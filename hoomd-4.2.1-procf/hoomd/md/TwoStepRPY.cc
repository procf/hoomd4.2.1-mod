//~ ########## Created by the Rheoinformatic research group ##########
//~ HOOMD-blue:
// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.
//~
//~ This file:
//~ Written by Dr. Mingyang Tan

/*! \file TwoStepRPY.h
    \brief Declares an integrator that updates the positions of particles while considering the hydrodynamic interactions (HI) with Rotne-Prager-Yamakawa tensor
*/

#include "TwoStepRPY.h"
#include "hoomd/HOOMDMath.h"
#include "hoomd/VectorMath.h"
#include "hoomd/RNGIdentifiers.h"
#include "hoomd/RandomNumbers.h"
#include "hoomd/GlobalArray.h"
#include <map>
#include <lapacke.h>
#include <cblas.h>

#ifdef ENABLE_MPI
#include "hoomd/HOOMDMPI.h"
#endif

using namespace std;

namespace hoomd
    {
namespace md
    {
inline bool check_pow2(unsigned int n)
    {
    while (n && n % 2 == 0)
        {
        n /= 2;
        }

    return (n == 1);
    };

inline unsigned int closest_pow2(unsigned int n)
    {
    unsigned int lower = 1 << (int) log2(n);
    unsigned int upper = lower << 1;

    if (upper - n > n - lower)
        {
        return lower;
        }
    else
        {
        return upper;
        }
    };

TwoStepRPY::TwoStepRPY(std::shared_ptr<SystemDefinition> sysdef,
                       std::shared_ptr<ParticleGroup> group,
                       std::shared_ptr<Variant> T,
                       std::shared_ptr<NeighborList> nlist,
                       Scalar xi,
                       Scalar error) : IntegrationMethodTwoStep(sysdef, group), m_T(T), m_nlist(nlist), m_xi(xi), m_error(error)
    {
    m_exec_conf->msg->notice(5) << "Constructing RPY\n";
    m_rcut_ewald = Scalar(0.0);

    m_mesh_points = make_uint3(0, 0, 0);
    m_global_dim = make_uint3(0, 0, 0);
    m_n_ghost_cells = make_uint3(0, 0, 0);
    m_grid_dim = make_uint3(0, 0, 0);
    m_ghost_width = make_scalar3(0, 0, 0);
    m_ghost_offset = 0;
    m_n_cells = 0;
    m_radius = 1;
    m_n_inner_cells = 0;

    m_box_changed = false;
    m_need_initialize = true;
    m_kiss_fft_initialized = false;
    m_dfft_initialized = false;

    m_pdata->getBoxChangeSignal().connect<TwoStepRPY, &TwoStepRPY::setBoxChange>(this);

    } /* end of constructor*/

void TwoStepRPY::setParams()
    {
    // find the cut-off radius for Ewald
    Scalar xi2 = m_xi * m_xi;
    Scalar xi3 = xi2 * m_xi;
    Scalar xi4 = xi2 * xi2;
    m_xi2 = xi2;
    m_pisqrt = fast::sqrt(M_PI);
    m_xi_pisqrt_inv = m_xi / m_pisqrt;

    const BoxDim& global_box = m_pdata->getGlobalBox();
    Scalar3 L = global_box.getL();

    m_rcut_ewald = fast::sqrt( - fast::log(m_error) ) / m_xi;
    if ( (m_rcut_ewald > L.x / 2.0) || (m_rcut_ewald > L.y / 2.0) || (m_rcut_ewald > L.z / 2.0) )
        {
        printf("Cut-off radius in Ewald Real Space is too large.\n");
        exit(EXIT_FAILURE);
        }
    
    // find the number of mesh points along each axis
    unsigned int kcut = (unsigned int) ( 2.0 * fast::sqrt( - fast::log(m_error) ) * m_xi ) + 1;
    m_mesh_points.x = (unsigned int) (kcut * L.x / M_PI) + 1;
    m_mesh_points.y = (unsigned int) (kcut * L.y / M_PI) + 1;
    m_mesh_points.z = (unsigned int) (kcut * L.z / M_PI) + 1;

    // The number of mesh points must be a power of 2
    m_mesh_points.x = closest_pow2(m_mesh_points.x);
    m_mesh_points.y = closest_pow2(m_mesh_points.y);
    m_mesh_points.z = closest_pow2(m_mesh_points.z);

    m_global_dim = m_mesh_points;
    m_NNN = m_global_dim.x * m_global_dim.y * m_global_dim.z;

#ifdef ENABLE_MPI
    if (m_pdata->getDomainDecomposition())
        {
        const Index3D& didx = m_pdata->getDomainDecomposition()->getDomainIndexer();
        if (!check_pow2(m_mesh_points.x) || !check_pow2(m_mesh_points.y) || !check_pow2(m_mesh_points.z))
            {
            throw std::runtime_error(
                "The number of mesh points along the every direction must be a power of two!");
            }

        if (m_mesh_points.x % didx.getW())
            {
            std::ostringstream s;
            s << "The number of mesh points along the x-direction (" << m_mesh_points.x << ") is not"
              << "a multiple of the width (" << didx.getW() << ") of the processor grid!";
            throw std::runtime_error(s.str());
            }
        if (m_mesh_points.y % didx.getH())
            {
            std::ostringstream s;
            s << "The number of mesh points along the y-direction (" << m_mesh_points.y << ") is not"
              << "a multiple of the height (" << didx.getH() << ") of the processor grid!";
            throw std::runtime_error(s.str());
            }
        if (m_mesh_points.z % didx.getD())
            {
            std::ostringstream s;
            s << "The number of mesh points along the z-direction (" << m_mesh_points.z << ") is not"
              << "a multiple of the depth (" << didx.getD() << ") of the processor grid!";
            throw std::runtime_error(s.str());
            }

        m_mesh_points.x /= didx.getW();
        m_mesh_points.y /= didx.getH();
        m_mesh_points.z /= didx.getD();
        }
    m_ghost_offset = 0;
#endif

    // size of mesh grid
    Scalar hx = L.x / (Scalar) m_global_dim.x;
    Scalar hy = L.y / (Scalar) m_global_dim.y;
    Scalar hz = L.z / (Scalar) m_global_dim.z;
    m_h = make_scalar3(hx, hy, hz);
    m_vk = hx * hy * hz;

    // Find the number of supporting points
    Scalar gamma = 0.5;
    Scalar gamma2 = gamma * gamma;
    Scalar lambda = 1.0 + 0.5 * gamma2 + gamma * fast::sqrt(1.0 + 0.25 * gamma2);
    Scalar gauss = Scalar(1.0);
    while ( fast::erfc( gauss / fast::sqrt(2.0 * lambda) ) > m_error )
        {
        gauss += 0.01;
        }
    m_P = (unsigned int) (gauss * gauss / M_PI) + 1;
    if (m_P % 2 == 0)
        {
        m_P += 1;
        }

    if (m_P > m_mesh_points.x)
        m_P = m_mesh_points.x;
    if (m_P > m_mesh_points.y)
        m_P = m_mesh_points.y;
    if (m_P > m_mesh_points.z)
        m_P = m_mesh_points.z;
    m_radius = m_P / 2;
    
    // Find the parameters for Spectral Ewald
    Scalar w = Scalar(m_P) * hx / 2.0;
    m_eta = (2.0 * w / gauss) * (2.0 * w / gauss) * xi2;
    m_gauss_fac = 2.0 * xi2 / (M_PI * m_eta) * fast::sqrt( 2.0 * xi2 / (M_PI * m_eta) );
    m_gauss_exp = 2.0 * xi2 / m_eta;
    
    m_self_func.x = 1.0 - m_xi / ( 3.0 * m_pisqrt ) * (9.0 - 10.0 * xi2 + 7.0 * xi4);
    m_self_func.y = 0.75 - xi3 / ( 10.0 * m_pisqrt ) * (25.0 - 42.0 * xi2 + 27.0 * xi4);
    
    m_need_initialize = true;
    std::cout << "Parameters setup finished\n" << std::endl;
    std::cout << "xi = " << m_xi << std::endl;
    std::cout << "rcut = " << m_rcut_ewald << std::endl;
    std::cout << "xa11 = " << m_self_func.x << ", xc11 = " << m_self_func.y << std::endl;
    std::cout << "global dimension: " << m_global_dim.x << ", " << m_global_dim.y << ", " << m_global_dim.z << std::endl;
    std::cout << "mesh dimension: " << m_mesh_points.x << ", " << m_mesh_points.y << ", " << m_mesh_points.z << std::endl;
    std::cout << "grid dimension: " << m_grid_dim.x << ", " << m_grid_dim.y << ", " << m_grid_dim.z << std::endl;
    std::cout << "ghost cell dimension: " << m_n_ghost_cells.x << ", " << m_n_ghost_cells.y << ", " << m_n_ghost_cells.z << std::endl;
    std::cout << "total mesh = " << m_NNN << std::endl;
    std::cout << "ghost offset = " << m_ghost_offset << std::endl;
    std::cout << "total inner mesh = " << m_n_cells << std::endl;
    std::cout << "inner mesh = " << m_n_inner_cells << std::endl;
    std::cout << "P = " << m_P << std::endl;
    std::cout << "eta = " << m_eta << std::endl;

    } /* end of setParams */

TwoStepRPY::~TwoStepRPY()
    {
    printf("Destroy RPY\n");
    m_exec_conf->msg->notice(5) << "Destroying RPY\n";

    if (m_kiss_fft_initialized)
        {
        kiss_fft_free(m_kiss_fft);
        kiss_fft_free(m_kiss_ifft);
        kiss_fft_cleanup();    
        }
#ifdef ENABLE_MPI
    if (m_dfft_initialized)
        {
        dfft_destroy_plan(m_dfft_plan_forward);
        dfft_destroy_plan(m_dfft_plan_inverse);
        }
#endif
    m_pdata->getBoxChangeSignal().disconnect<TwoStepRPY, &TwoStepRPY::setBoxChange>(this);
    } /* End of ~TwoStepRPY */

void TwoStepRPY::setupMesh()
    {
    // update number of ghost cells
    m_n_ghost_cells = computeGhostCellNum();

    // extra ghost cells are as wide as the inner cells
    const BoxDim& box = m_pdata->getBox();
    Scalar3 cell_width = box.getNearestPlaneDistance()
                         / make_scalar3(m_mesh_points.x, m_mesh_points.y, m_mesh_points.z);
    m_ghost_width
        = cell_width * make_scalar3(m_n_ghost_cells.x, m_n_ghost_cells.y, m_n_ghost_cells.z);

    m_exec_conf->msg->notice(6) << "RPY: (Re-)allocating ghost layer (" << m_n_ghost_cells.x
                                << "," << m_n_ghost_cells.y << "," << m_n_ghost_cells.z << ")"
                                << std::endl;
    
    m_grid_dim = make_uint3(m_mesh_points.x + 2 * m_n_ghost_cells.x,
                            m_mesh_points.y + 2 * m_n_ghost_cells.y,
                            m_mesh_points.z + 2 * m_n_ghost_cells.z);
    m_n_cells = m_grid_dim.x * m_grid_dim.y * m_grid_dim.z;
    m_n_inner_cells = m_mesh_points.x * m_mesh_points.y * m_mesh_points.z;

    // allocate memory for influence function and k values
    GlobalArray<Scalar3> ymob(m_n_inner_cells, m_exec_conf);
    m_ymob.swap(ymob);

    GlobalArray<Scalar4> gridk(m_n_inner_cells, m_exec_conf);
    m_gridk.swap(gridk);

    GlobalArray<Scalar2> sf(m_n_inner_cells, m_exec_conf);
    m_sf.swap(sf);

    initializeFFT();
    } /* end of setupMesh */

uint3 TwoStepRPY::computeGhostCellNum()
    {
    // ghost cells
    uint3 n_ghost_cells = make_uint3(0, 0, 0);
#ifdef ENABLE_MPI
    if (m_pdata->getDomainDecomposition())
        {
        Index3D di = m_pdata->getDomainDecomposition()->getDomainIndexer();
        n_ghost_cells.x = (di.getW() > 1) ? m_radius : 0;
        n_ghost_cells.y = (di.getH() > 1) ? m_radius : 0;
        n_ghost_cells.z = (di.getD() > 1) ? m_radius : 0;
        }
#endif

    // extra ghost cells to accommodate skin layer (max 1/2 ghost layer width)
#ifdef ENABLE_MPI
    if (m_sysdef->isDomainDecomposed())
        {
        Scalar r_buff = m_nlist->getRBuff() / 2.0;

        const BoxDim& box = m_pdata->getBox();
        Scalar3 cell_width = box.getNearestPlaneDistance()
                             / make_scalar3(m_mesh_points.x, m_mesh_points.y, m_mesh_points.z);

        if (n_ghost_cells.x)
            n_ghost_cells.x += (unsigned int)(r_buff / cell_width.x) + 1;
        if (n_ghost_cells.y)
            n_ghost_cells.y += (unsigned int)(r_buff / cell_width.y) + 1;
        if (n_ghost_cells.z)
            n_ghost_cells.z += (unsigned int)(r_buff / cell_width.z) + 1;
        }
#endif
    return n_ghost_cells;
    } /* End of computeGhostCellNum */

void TwoStepRPY::initializeFFT()
    {
    bool local_fft = true;
#ifdef ENABLE_MPI
    local_fft = !m_pdata->getDomainDecomposition();
    if (!local_fft)
        {
        // ghost cell communicator for force interpolation
        m_grid_comm_forward
            = std::unique_ptr<CommunicatorGrid<kiss_fft_cpx>>(new CommunicatorGrid<kiss_fft_cpx>(
                m_sysdef,
                make_uint3(m_mesh_points.x, m_mesh_points.y, m_mesh_points.z),
                make_uint3(m_grid_dim.x, m_grid_dim.y, m_grid_dim.z),
                m_n_ghost_cells,
                true));
        // ghost cell communicator for force mesh
        m_grid_comm_reverse
            = std::unique_ptr<CommunicatorGrid<kiss_fft_cpx>>(new CommunicatorGrid<kiss_fft_cpx>(
                m_sysdef,
                make_uint3(m_mesh_points.x, m_mesh_points.y, m_mesh_points.z),
                make_uint3(m_grid_dim.x, m_grid_dim.y, m_grid_dim.z),
                m_n_ghost_cells,
                false));
        // set up distributed FFTs
        int gdim[3];
        int pdim[3];
        Index3D decomp_idx = m_pdata->getDomainDecomposition()->getDomainIndexer();
        pdim[0] = decomp_idx.getD();
        pdim[1] = decomp_idx.getH();
        pdim[2] = decomp_idx.getW();
        gdim[0] = m_mesh_points.z * pdim[0];
        gdim[1] = m_mesh_points.y * pdim[1];
        gdim[2] = m_mesh_points.x * pdim[2];
        int embed[3];
        embed[0] = m_mesh_points.z + 2 * m_n_ghost_cells.z;
        embed[1] = m_mesh_points.y + 2 * m_n_ghost_cells.y;
        embed[2] = m_mesh_points.x + 2 * m_n_ghost_cells.x;
        m_ghost_offset
            = (m_n_ghost_cells.z * embed[1] + m_n_ghost_cells.y) * embed[2] + m_n_ghost_cells.x;
        uint3 pcoord = m_pdata->getDomainDecomposition()->getGridPos();
        int pidx[3];
        pidx[0] = pcoord.z;
        pidx[1] = pcoord.y;
        pidx[2] = pcoord.x;
        int row_m = 0; /* both local grid and proc grid are row major, no transposition necessary */
        ArrayHandle<unsigned int> h_cart_ranks(m_pdata->getDomainDecomposition()->getCartRanks(),
                                               access_location::host,
                                               access_mode::read);
        dfft_create_plan(&m_dfft_plan_forward,
                         3,
                         gdim,
                         embed,
                         NULL,
                         pdim,
                         pidx,
                         row_m,
                         0,
                         1,
                         m_exec_conf->getMPICommunicator(),
                         (int*)h_cart_ranks.data);
        dfft_create_plan(&m_dfft_plan_inverse,
                         3,
                         gdim,
                         NULL,
                         embed,
                         pdim,
                         pidx,
                         row_m,
                         0,
                         1,
                         m_exec_conf->getMPICommunicator(),
                         (int*)h_cart_ranks.data);
        m_dfft_initialized = true;
        } // if !local_fft
#endif

    if (local_fft)
        {
        int dims[3];
        dims[0] = m_mesh_points.z;
        dims[1] = m_mesh_points.y;
        dims[2] = m_mesh_points.x;

        if (m_kiss_fft)
            {
            kiss_fft_free(m_kiss_fft);
            }
        if (m_kiss_ifft)
            {
            kiss_fft_free(m_kiss_ifft);
            }
        
        m_kiss_fft = kiss_fftnd_alloc(dims, 3, 0, NULL, NULL);
        m_kiss_ifft = kiss_fftnd_alloc(dims, 3, 1, NULL, NULL);

        m_kiss_fft_initialized = true;
        }

    // allocate memory for mesh (padded with offset)
    unsigned int mesh_size = m_n_cells + m_ghost_offset;
    
    GlobalArray<kiss_fft_cpx> mesh_Fx(mesh_size, m_exec_conf);
    m_mesh_Fx.swap(mesh_Fx);

    GlobalArray<kiss_fft_cpx> mesh_Fy(mesh_size, m_exec_conf);
    m_mesh_Fy.swap(mesh_Fy);

    GlobalArray<kiss_fft_cpx> mesh_Fz(mesh_size, m_exec_conf);
    m_mesh_Fz.swap(mesh_Fz);


    // allocate memory for FFT mesh
    unsigned int fft_mesh_size = m_n_inner_cells;

    GlobalArray<kiss_fft_cpx> mesh_fft_Fx(fft_mesh_size, m_exec_conf);
    m_mesh_fft_Fx.swap(mesh_fft_Fx);

    GlobalArray<kiss_fft_cpx> mesh_fft_Fy(fft_mesh_size, m_exec_conf);
    m_mesh_fft_Fy.swap(mesh_fft_Fy);

    GlobalArray<kiss_fft_cpx> mesh_fft_Fz(fft_mesh_size, m_exec_conf);
    m_mesh_fft_Fz.swap(mesh_fft_Fz);

    // allocate memory for IFFT mesh (padded with offset)
    GlobalArray<kiss_fft_cpx> mesh_inv_Fx(mesh_size, m_exec_conf);
    m_mesh_inv_Fx.swap(mesh_inv_Fx);

    GlobalArray<kiss_fft_cpx> mesh_inv_Fy(mesh_size, m_exec_conf);
    m_mesh_inv_Fy.swap(mesh_inv_Fy);

    GlobalArray<kiss_fft_cpx> mesh_inv_Fz(mesh_size, m_exec_conf);
    m_mesh_inv_Fz.swap(mesh_inv_Fz);

    } /* end of initializeFFT */

void TwoStepRPY::initializeWorkArray()
    {
    m_m_lanczos_ff = 2;
    m_mmax = 100;
    unsigned int group_size = m_group->getNumMembers();
    
    // size for Brownian vectors
    unsigned int numel = group_size * 3;
    GlobalArray<Scalar> uslip(numel, m_exec_conf);
    m_uslip.swap(uslip);

    GlobalArray<Scalar> uslip_wave(numel, m_exec_conf);
    m_uslip_wave.swap(uslip_wave);

    // size for Lanczos iterations
    GlobalArray<Scalar> iter_ff_v(numel, m_exec_conf);
    m_iter_ff_v.swap(iter_ff_v);

    GlobalArray<Scalar> iter_ff_vj(numel, m_exec_conf);
    m_iter_ff_vj.swap(iter_ff_vj);

    GlobalArray<Scalar> iter_ff_vjm1(numel, m_exec_conf);
    m_iter_ff_vjm1.swap(iter_ff_vjm1);

    GlobalArray<Scalar> iter_ff_V(numel * (m_mmax + 1), m_exec_conf);
    m_iter_ff_V.swap(iter_ff_V);

    GlobalArray<Scalar> iter_ff_uold(numel, m_exec_conf);
    m_iter_ff_uold.swap(iter_ff_uold);

    GlobalArray<Scalar> iter_ff_u(numel, m_exec_conf);
    m_iter_ff_u.swap(iter_ff_u);

    GlobalArray<Scalar> iter_psi(numel, m_exec_conf);
    m_iter_psi.swap(iter_psi);

    GlobalArray<Scalar> iter_Mpsi(numel, m_exec_conf);
    m_iter_Mpsi.swap(iter_Mpsi);

    // size of M . F = U calculation
    GlobalArray<Scalar> u_real(numel, m_exec_conf);
    m_u_real.swap(u_real);

    GlobalArray<Scalar> u_wave(numel, m_exec_conf);
    m_u_wave.swap(u_wave);

    GlobalArray<Scalar> u_determin(numel, m_exec_conf);
    m_u_determin.swap(u_determin);

    // size of velocity and force
    GlobalArray<Scalar> uoe(numel, m_exec_conf);
    m_uoe.swap(uoe);

    GlobalArray<Scalar> fts(numel, m_exec_conf);
    m_fts.swap(fts);

    GlobalArray<Scalar> mobmat_real(numel * numel, m_exec_conf);
    m_mobmat_real.swap(mobmat_real);

    GlobalArray<Scalar> mobmat_wave(numel * numel, m_exec_conf);
    m_mobmat_wave.swap(mobmat_wave);

    GlobalArray<Scalar> mobmat(numel * numel, m_exec_conf);
    m_mobmat.swap(mobmat);

    GlobalArray<Scalar> mobmat_scratch(numel * numel, m_exec_conf);
    m_mobmat_scratch.swap(mobmat_scratch);

    GlobalArray<Scalar> mobmatSqrt(numel * numel, m_exec_conf);
    m_mobmatSqrt.swap(mobmatSqrt);

    GlobalArray<Scalar> kron(9, m_exec_conf);
    m_kron.swap(kron);

    } /* end of initializeWorkArray */

void TwoStepRPY::computeWaveValue()
    {
    ArrayHandle<Scalar4> h_gridk(m_gridk, 
                                 access_location::host, 
                                 access_mode::overwrite);

    ArrayHandle<Scalar3> h_ymob(m_ymob,
                                access_location::host, 
                                access_mode::overwrite);
    
    ArrayHandle<Scalar2> h_sf(m_sf, 
                              access_location::host, 
                              access_mode::overwrite);

    // zero arrays
    memset(h_gridk.data, 0, sizeof(Scalar4) * m_gridk.getNumElements());
    memset(h_ymob.data, 0, sizeof(Scalar3) * m_ymob.getNumElements());
    memset(h_sf.data, 0, sizeof(Scalar2) * m_sf.getNumElements());
    
    const BoxDim& global_box = m_pdata->getGlobalBox();

    // compute reciprocal lattice vectors
    Scalar3 a1 = global_box.getLatticeVector(0);
    Scalar3 a2 = global_box.getLatticeVector(1);
    Scalar3 a3 = global_box.getLatticeVector(2);
    Scalar V_box = global_box.getVolume();

    Scalar3 b1 = Scalar(2.0 * M_PI)
                 * make_scalar3(a2.y * a3.z - a2.z * a3.y,
                                a2.z * a3.x - a2.x * a3.z,
                                a2.x * a3.y - a2.y * a3.x)
                 / V_box;
    Scalar3 b2 = Scalar(2.0 * M_PI)
                 * make_scalar3(a3.y * a1.z - a3.z * a1.y,
                                a3.z * a1.x - a3.x * a1.z,
                                a3.x * a1.y - a3.y * a1.x)
                 / V_box;
    Scalar3 b3 = Scalar(2.0 * M_PI)
                 * make_scalar3(a1.y * a2.z - a1.z * a2.y,
                                a1.z * a2.x - a1.x * a2.z,
                                a1.x * a2.y - a1.y * a2.x)
                 / V_box;

#ifdef ENABLE_MPI
    bool local_fft = m_kiss_fft_initialized;

    uint3 pdim = make_uint3(0, 0, 0);
    uint3 pidx = make_uint3(0, 0, 0);
    if (m_pdata->getDomainDecomposition())
        {
        const Index3D& didx = m_pdata->getDomainDecomposition()->getDomainIndexer();
        pidx = m_pdata->getDomainDecomposition()->getGridPos();
        pdim = make_uint3(didx.getW(), didx.getH(), didx.getD());
        }
#endif

    for (unsigned int cell_idx = 0; cell_idx < m_n_inner_cells; cell_idx++)
        {
        uint3 wave_idx;
#ifdef ENABLE_MPI
        if (!local_fft)
            {
            // local layout: row major
            int ny = m_mesh_points.y;
            int nx = m_mesh_points.x;
            int n_local = cell_idx / ny / nx;
            int m_local = (cell_idx - n_local * ny * nx) / nx;
            int l_local = cell_idx % nx;
            // cyclic distribution
            wave_idx.x = l_local * pdim.x + pidx.x;
            wave_idx.y = m_local * pdim.y + pidx.y;
            wave_idx.z = n_local * pdim.z + pidx.z;
            }
        else
#endif
            {
            // kiss FFT expects data in row major format
            wave_idx.z = cell_idx / (m_mesh_points.y * m_mesh_points.x);
            wave_idx.y
                = (cell_idx - wave_idx.z * m_mesh_points.x * m_mesh_points.y) / m_mesh_points.x;
            wave_idx.x = cell_idx % m_mesh_points.x;
            }
    
        int3 n = make_int3(wave_idx.x, wave_idx.y, wave_idx.z);
        // compute Miller indices
        if (n.x >= (int)(m_global_dim.x / 2 + m_global_dim.x % 2))
            n.x -= (int)m_global_dim.x;
        if (n.y >= (int)(m_global_dim.y / 2 + m_global_dim.y % 2))
            n.y -= (int)m_global_dim.y;
        if (n.z >= (int)(m_global_dim.z / 2 + m_global_dim.z % 2))
            n.z -= (int)m_global_dim.z;
        
        Scalar kx = Scalar(n.x) * b1.x + Scalar(n.y) * b2.x + Scalar(n.z) * b3.x;
        Scalar ky = Scalar(n.x) * b1.y + Scalar(n.y) * b2.y + Scalar(n.z) * b3.y;
        Scalar kz = Scalar(n.x) * b1.z + Scalar(n.y) * b2.z + Scalar(n.z) * b3.z;
        Scalar kk = kx * kx + ky * ky + kz * kz;
        
        Scalar ks = sqrt(kk);

        if (n.x != 0 || n.y != 0 || n.z != 0)
            {
            h_gridk.data[cell_idx] = make_scalar4(kx / ks, ky / ks, kz / ks, ks);
            Scalar kxi = kk / (4.0 * m_xi2);
            Scalar ya = 6.0 * M_PI * (1.0 + kxi) / kk * exp( - (1.0 - m_eta) * kxi) / (Scalar(m_NNN));
            Scalar yb = 0.5 * ks * ya;
            Scalar yc = 0.25 * kk * ya;

            Scalar sf_uf = 1.0 - kk / 6.0;
            Scalar sf_ot = 1.0 - kk / 10.0;

            h_ymob.data[cell_idx].x = ya;
            h_ymob.data[cell_idx].y = yb;
            h_ymob.data[cell_idx].z = yc;

            h_sf.data[cell_idx].x = sf_uf;
            h_sf.data[cell_idx].y = sf_ot;
            }
        else
            {
            h_gridk.data[cell_idx] = make_scalar4(0, 0, 0, 0);
            h_ymob.data[cell_idx] = make_scalar3(0.0, 0.0, 0.0);
            h_sf.data[cell_idx] = make_scalar2(0.0, 0.0);
            }
        
        } // for cell_idx

    } /* end of computeWaveValue */

/*! Computes real-space mobility functions

    r           input   distance between 2 particles
    xa, ya      output  UF coupling functions
    yb          output  UT or OF coupling function
    xc, yc      output  OT coupling functions  
 */
void TwoStepRPY::mobilityRealFunc(Scalar r,
                                  Scalar& xa, Scalar& ya,
                                  Scalar& yb,
                                  Scalar& xc, Scalar &yc)
    {
    Scalar xir = m_xi * r;
    Scalar xir2 = xir * xir;
    Scalar expxir2 = fast::exp(-xir2);

    Scalar phi0 = m_pisqrt / xir * fast::erfc(xir);
    Scalar phi1 = Scalar(1.0) / xir2 * (Scalar(0.5) * phi0 + expxir2);
    Scalar phi2 = Scalar(1.0) / xir2 * (Scalar(1.5) * phi1 + expxir2);
    Scalar phi3 = Scalar(1.0) / xir2 * (Scalar(2.5) * phi2 + expxir2);
    Scalar phi4 = Scalar(1.0) / xir2 * (Scalar(3.5) * phi3 + expxir2);
    Scalar phi5 = Scalar(1.0) / xir2 * (Scalar(4.5) * phi4 + expxir2);
    Scalar phi6 = Scalar(1.0) / xir2 * (Scalar(5.5) * phi5 + expxir2);
    Scalar phi7 = Scalar(1.0) / xir2 * (Scalar(6.5) * phi6 + expxir2);

    xa = 0.0;
    ya = 0.0;
    yb = 0.0;
    xc = 0.0;
    yc = 0.0;

    xa = m_xi_pisqrt_inv * 
                    (
                    Scalar(2.0) * phi0
                    + m_xi2 / Scalar(3.0) * ( - Scalar(20.0) * phi1 + Scalar(8.0) * xir2 * phi2
                                    + m_xi2 / Scalar(3.0) * (
                                                    Scalar(70.0) * phi2 + xir2 * ( - Scalar(56.0) * phi3 + xir2 * (Scalar(8.0) * phi4) )
                                                )
                                    )
                    );
    ya = m_xi_pisqrt_inv * 
                    (
                    Scalar(2.0) * phi0 + xir2 * ( - Scalar(2.0) * phi1)
                    + m_xi2 / Scalar(3.0) * (
                                    - Scalar(20.0) * phi1 + xir2 * (Scalar(36.0) * phi2 + xir2 * ( - Scalar(8.0) * phi3))
                                    + m_xi2 / Scalar(3.0) * (
                                                    Scalar(70.0) * phi2 + xir2 * ( - Scalar(182.0) * phi3 + xir2 * (Scalar(80.0) * phi4 - xir2 * (Scalar(8.0) * phi5)) )
                                                )
                                    )
                    );
    xa *= Scalar(0.75);
    ya *= Scalar(0.75);

    yb = m_xi_pisqrt_inv * r * m_xi2 * 
                (
                - Scalar(5.0) * phi1 + xir2 * (Scalar(2.0) * phi2)
                + m_xi2 / Scalar(15.0) * (
                                Scalar(280.0) * phi2 + xir2 * (- Scalar(224.0) * phi3 + xir2 * (Scalar(32.0) * phi4))
                                + m_xi2 * (
                                            - Scalar(315.0) * phi3 + xir2 * (Scalar(378.0) * phi4 + xir2 * (- Scalar(108.0) * phi5 + xir2 * (Scalar(8.0) * phi6)))
                                        )
                                )
                );

    xc = m_xi_pisqrt_inv * m_xi2 * 
                    (
                    Scalar(5.0) * phi1 + xir2 * (- Scalar(2.0) * phi2)
                    + m_xi2 / Scalar(5.0) * ( 
                                            - Scalar(70.0) * phi2 + xir2 * (Scalar(56.0) * phi3 + xir2 * (- Scalar(8.0) * phi4))
                                            + m_xi2 / Scalar(5.0) * (
                                                        Scalar(315.0) * phi3 + xir2 * ( - Scalar(378.0) * phi4 + xir2 * (Scalar(108.0) * phi5 + xir2 * (- Scalar(8.0) * phi6)))
                                                        )
                                        )
                    );
    yc = m_xi_pisqrt_inv * m_xi2 * 
                    (
                    Scalar(5.0) * phi1 + xir2 * (- Scalar(9.0) * phi2 + xir2 * (Scalar(2.0) * phi3))
                    + m_xi2 / Scalar(5.0) * ( 
                                            - Scalar(70.0) * phi2 + xir2 * (Scalar(182.0) * phi3 + xir2 * (- Scalar(80.0) * phi4 + xir2 * (Scalar(8.0) * phi5)))
                                            + m_xi2 / Scalar(5.0) * (
                                                        Scalar(315.0) * phi3 + xir2 * ( - Scalar(1071.0) * phi4 + xir2 * (Scalar(702.0) * phi5 + xir2 * (- Scalar(140.0) * phi6 + xir2 * (Scalar(8.0) * phi7))))
                                                        )
                                        )
                    );
    yb *= Scalar(0.75);
    xc *= Scalar(0.75);
    yc *= Scalar(0.75);

    } /* end of mobilityRealFunc */

/*! Calculate the matrix-vector multiplication M . F in real space
    where F may contain force, torque, and stresslet

    fts         input   force/torque/stresslet
    ureal       output  real-space velocity/rotational velocity/rate of strain
 */
void TwoStepRPY::mobilityRealUF(Scalar * force,
                                Scalar * ureal)
    {
    unsigned int group_size = m_group->getNumMembers();
    const BoxDim& box = m_pdata->getGlobalBox();
    Scalar rcut = m_rcut_ewald;

    ArrayHandle<unsigned int> h_nneigh(m_nlist->getNNeighArray(),
                                       access_location::host,
                                       access_mode::read);
    ArrayHandle<unsigned int> h_nlist(m_nlist->getNListArray(),
                                      access_location::host,
                                      access_mode::read);
    ArrayHandle<size_t> h_headlist(m_nlist->getHeadList(),
                                   access_location::host,
                                   access_mode::read);

    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(),
                               access_location::host,
                               access_mode::read);

    Scalar3 Fi = make_scalar3(0.0, 0.0, 0.0);
    Scalar3 Fj = make_scalar3(0.0, 0.0, 0.0);
    
    for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
        {
        unsigned int idx = m_group->getMemberIndex(group_idx);
        unsigned int i3 = idx * 3;

        Scalar3 U = make_scalar3(0.0, 0.0, 0.0);        // translational velocity

        Scalar3 posi = make_scalar3(h_pos.data[idx].x, h_pos.data[idx].y, h_pos.data[idx].z);

        Fi.x = force[i3    ];
        Fi.y = force[i3 + 1];
        Fi.z = force[i3 + 2];

        // U induced by F of particle i
        U.x = m_self_func.x * Fi.x;
        U.y = m_self_func.x * Fi.y;
        U.z = m_self_func.x * Fi.z;

        unsigned int nneigh_i = h_nneigh.data[idx];
        size_t head_i = h_headlist.data[idx];

        for (unsigned int neigh_idx = 0; neigh_idx < nneigh_i; neigh_idx++)
            {
            unsigned int jdx = h_nlist.data[head_i + neigh_idx];
            unsigned int j3 = jdx * 3;

            Scalar3 posj = make_scalar3(h_pos.data[jdx].x, h_pos.data[jdx].y, h_pos.data[jdx].z);
            Fj.x = force[j3    ];
            Fj.y = force[j3 + 1];
            Fj.z = force[j3 + 2];
            
            Scalar3 dist = posj - posi;
            dist = box.minImage(dist);
            Scalar r2 = dot(dist, dist);
            Scalar r = fast::sqrt(r2);

            if (r < rcut)
                {
                Scalar3 e = dist / r;

                if (r < 2.0)
                    {
                    r = 2.0;
                    }

                Scalar xa, ya, yb, xc, yc;
                mobilityRealFunc(r,
                                 xa, ya,
                                 yb,
                                 xc, yc);

                // UF coupling 
                // U = ya * \mathbf{F} + (xa - ya) * (\mathbf{F} \cdot \mathbf{e}) * \mathbf{e}
                Scalar xmya = xa - ya;
                Scalar Fj_dot_e = dot(Fj, e);
                U.x += ya * Fj.x + xmya * Fj_dot_e * e.x;
                U.y += ya * Fj.y + xmya * Fj_dot_e * e.y;
                U.z += ya * Fj.z + xmya * Fj_dot_e * e.z;
                } // if r < rcut_ewald

            } // for neigh_idx
        
        // add velocity to the output array
        ureal[i3    ] = U.x;
        ureal[i3 + 1] = U.y;
        ureal[i3 + 2] = U.z;
        } // for group_idx
    
    } /* end of mobilityRealUF*/

/*! Assign force moments of particles to mesh
    
    fts     input   force/torque/stresslet of particles
 */
void TwoStepRPY::assignParticleForce(Scalar * force)
    {
    unsigned int group_size = m_group->getNumMembers();
    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(),
                               access_location::host,
                               access_mode::read);
    const BoxDim& box = m_pdata->getBox();
    
    // access mesh data
    ArrayHandle<kiss_fft_cpx> h_mesh_Fx(m_mesh_Fx,
                                        access_location::host,
                                        access_mode::readwrite);
    ArrayHandle<kiss_fft_cpx> h_mesh_Fy(m_mesh_Fy,
                                        access_location::host,
                                        access_mode::readwrite);
    ArrayHandle<kiss_fft_cpx> h_mesh_Fz(m_mesh_Fz,
                                        access_location::host,
                                        access_mode::readwrite);
    
    // zero mesh
    memset(h_mesh_Fx.data, 0, sizeof(kiss_fft_cpx) * m_mesh_Fx.getNumElements());
    memset(h_mesh_Fy.data, 0, sizeof(kiss_fft_cpx) * m_mesh_Fy.getNumElements());
    memset(h_mesh_Fz.data, 0, sizeof(kiss_fft_cpx) * m_mesh_Fz.getNumElements());

    // loop over particles
    for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
        {
        unsigned int idx = m_group->getMemberIndex(group_idx);
        Scalar3 pos = make_scalar3(h_pos.data[idx].x, h_pos.data[idx].y, h_pos.data[idx].z);

        unsigned int i3 = idx * 3;
        Scalar3 F = make_scalar3(force[i3    ], force[i3 + 1], force[i3 + 2]);

        Scalar3 fpos = box.makeFraction(pos);
        Scalar3 reduced_pos = make_scalar3(fpos.x * (Scalar)m_mesh_points.x, 
                                           fpos.y * (Scalar)m_mesh_points.y,
                                           fpos.z * (Scalar)m_mesh_points.z);

        reduced_pos.x += (Scalar)m_n_ghost_cells.x;
        reduced_pos.y += (Scalar)m_n_ghost_cells.y;
        reduced_pos.z += (Scalar)m_n_ghost_cells.z;

        Scalar shift, shiftone;

        if (m_P % 2)
            {
            shift = 0.5;
            shiftone = 0.0;
            }
        else
            {
            shift = 0.0;
            shiftone = 0.5;
            }

        int ix = int(reduced_pos.x + shift);
        int iy = int(reduced_pos.y + shift);
        int iz = int(reduced_pos.z + shift);

        Scalar dx = shiftone + (Scalar)ix - reduced_pos.x;
        Scalar dy = shiftone + (Scalar)iy - reduced_pos.y;
        Scalar dz = shiftone + (Scalar)iz - reduced_pos.z;

        // handle particles on the boundary
        if (ix == (int)m_grid_dim.x && !m_n_ghost_cells.x)
            ix = 0;
        if (iy == (int)m_grid_dim.y && !m_n_ghost_cells.y)
            iy = 0;
        if (iz == (int)m_grid_dim.z && !m_n_ghost_cells.z)
            iz = 0;

        if (ix < 0 || ix >= (int)m_grid_dim.x || iy < 0 || iy >= (int)m_grid_dim.y || iz < 0
            || iz >= (int)m_grid_dim.z)
            {
            // ignore, error will be thrown elsewhere (in CellList)
            continue;
            }

        int nlower = -(static_cast<int>(m_P) - 1) / 2;
        int nupper = m_P / 2;
        // printf("nlower = %d, nupper = %d\n", nlower, nupper);

        for (int i = nlower; i <= nupper; ++i)
            {
            int neighi = (int)ix + i;
            Scalar rx = (Scalar(i) + dx) * m_h.x;

            if (!m_n_ghost_cells.x)
                {
                if (neighi >= (int)m_grid_dim.x)
                    neighi -= m_grid_dim.x;
                else if (neighi < 0)
                    neighi += m_grid_dim.x;
                }

            for (int j = nlower; j <= nupper; ++j)
                {
                int neighj = (int)iy + j;
                Scalar ry = (Scalar(j) + dy) * m_h.y;

                if (!m_n_ghost_cells.y)
                    {
                    if (neighj >= (int)m_grid_dim.y)
                        neighj -= m_grid_dim.y;
                    else if (neighj < 0)
                        neighj += m_grid_dim.y;
                    }
                
                for (int k = nlower; k <= nupper; ++k)
                    {
                    int neighk = (int)iz + k;
                    Scalar rz = (Scalar(k) + dz) * m_h.z;
                    
                    if (!m_n_ghost_cells.z)
                        {
                        if (neighk >= (int)m_grid_dim.z)
                            neighk -= m_grid_dim.z;
                        else if (neighk < 0)
                            neighk += m_grid_dim.z;
                        }

                    // store in row major order
                    unsigned int neigh_idx = neighi + m_grid_dim.x * (neighj + m_grid_dim.y * neighk);
                    Scalar r2 = rx * rx + ry * ry + rz * rz;
                    Scalar fac = m_gauss_fac * fast::exp( - m_gauss_exp * r2 );

                    // printf("Particle %u: cell %u, r2 = %f, xfac = %f, yfac = %f, zfac = %f\n", idx, neigh_idx, r2, F.x * fac, F.y * fac, F.z * fac);

                    h_mesh_Fx.data[neigh_idx].r += float(F.x * fac);
                    h_mesh_Fy.data[neigh_idx].r += float(F.y * fac);
                    h_mesh_Fz.data[neigh_idx].r += float(F.z * fac);
                    } // for k
                
                } // for j
            
            } // for i

        } // for group_idx
    
    } /* end of assignParticleForce */

/*! Conduct forward FFT of the force moments of mesh
 */
void TwoStepRPY::forwardFFT()
    {
    if (m_kiss_fft_initialized)
        {
        // forward FFT of mesh force
        // Fx
        ArrayHandle<kiss_fft_cpx> h_mesh_Fx(m_mesh_Fx, 
                                            access_location::host, 
                                            access_mode::read);
        ArrayHandle<kiss_fft_cpx> h_mesh_fft_Fx(m_mesh_fft_Fx, 
                                                access_location::host, 
                                                access_mode::overwrite);
        kiss_fftnd(m_kiss_fft, 
                   h_mesh_Fx.data, 
                   h_mesh_fft_Fx.data);
        // Fy
        ArrayHandle<kiss_fft_cpx> h_mesh_Fy(m_mesh_Fy, 
                                            access_location::host, 
                                            access_mode::read);
        ArrayHandle<kiss_fft_cpx> h_mesh_fft_Fy(m_mesh_fft_Fy, 
                                                access_location::host, 
                                                access_mode::overwrite);
        kiss_fftnd(m_kiss_fft, 
                   h_mesh_Fy.data, 
                   h_mesh_fft_Fy.data);
        // Fz
        ArrayHandle<kiss_fft_cpx> h_mesh_Fz(m_mesh_Fz, 
                                            access_location::host, 
                                            access_mode::read);
        ArrayHandle<kiss_fft_cpx> h_mesh_fft_Fz(m_mesh_fft_Fz, 
                                                access_location::host, 
                                                access_mode::overwrite);
        kiss_fftnd(m_kiss_fft,
                   h_mesh_Fz.data, 
                   h_mesh_fft_Fz.data);
        } // if m_kiss_fft_initialized (single CPU FFT)

#ifdef ENABLE_MPI
    if (m_pdata->getDomainDecomposition())
        {
        // update inner cells of particle mesh
        m_exec_conf->msg->notice(8) << "RPY: Ghost cell update" << std::endl;
        // perform a distributed FFT
        m_exec_conf->msg->notice(8) << "RPY: Distributed FFT mesh" << std::endl;
        m_grid_comm_forward->communicate(m_mesh_Fx);
        // Fx
        ArrayHandle<kiss_fft_cpx> h_mesh_Fx(m_mesh_Fx, 
                                            access_location::host, 
                                            access_mode::read);
        ArrayHandle<kiss_fft_cpx> h_mesh_fft_Fx(m_mesh_fft_Fx, 
                                                access_location::host, 
                                                access_mode::overwrite);
        dfft_execute((cpx_t*)(h_mesh_Fx.data + m_ghost_offset),
                     (cpx_t*)h_mesh_fft_Fx.data,
                     0,
                     m_dfft_plan_forward);

        m_grid_comm_forward->communicate(m_mesh_Fy);
        // Fy
        ArrayHandle<kiss_fft_cpx> h_mesh_Fy(m_mesh_Fy, 
                                            access_location::host, 
                                            access_mode::read);
        ArrayHandle<kiss_fft_cpx> h_mesh_fft_Fy(m_mesh_fft_Fy, 
                                                access_location::host, 
                                                access_mode::overwrite);
        dfft_execute((cpx_t*)(h_mesh_Fy.data + m_ghost_offset),
                     (cpx_t*)h_mesh_fft_Fy.data,
                     0,
                     m_dfft_plan_forward);

        m_grid_comm_forward->communicate(m_mesh_Fz);
        // Fz
        ArrayHandle<kiss_fft_cpx> h_mesh_Fz(m_mesh_Fz, 
                                            access_location::host, 
                                            access_mode::read);
        ArrayHandle<kiss_fft_cpx> h_mesh_fft_Fz(m_mesh_fft_Fz, 
                                                access_location::host, 
                                                access_mode::overwrite);
        dfft_execute((cpx_t*)(h_mesh_Fz.data + m_ghost_offset),
                     (cpx_t*)h_mesh_fft_Fz.data,
                     0,
                     m_dfft_plan_forward);
        } // if getDomainDecomposition
#endif

    } /* end of forwardFFT */

/*! Apply Green's function to FFTed force moments of mesh to get velocity moments of mesh in Fourier space
 */
void TwoStepRPY::meshGreen()
    {
    ArrayHandle<Scalar4> h_gridk(m_gridk, 
                                 access_location::host, 
                                 access_mode::read);
    ArrayHandle<Scalar3> h_ymob(m_ymob,
                                access_location::host,
                                access_mode::read);
    ArrayHandle<Scalar2> h_sf(m_sf,
                              access_location::host,
                              access_mode::read);
    // force of mesh                              
    ArrayHandle<kiss_fft_cpx> h_mesh_fft_Fx(m_mesh_fft_Fx,
                                            access_location::host, 
                                            access_mode::readwrite);
    ArrayHandle<kiss_fft_cpx> h_mesh_fft_Fy(m_mesh_fft_Fy,
                                            access_location::host, 
                                            access_mode::readwrite);
    ArrayHandle<kiss_fft_cpx> h_mesh_fft_Fz(m_mesh_fft_Fz,
                                            access_location::host, 
                                            access_mode::readwrite);

    kiss_fft_cpx Fx, Fy, Fz;
    for (unsigned int i = 0; i < m_n_inner_cells; i++)
        {
        Scalar kx = h_gridk.data[i].x;
        Scalar ky = h_gridk.data[i].y;
        Scalar kz = h_gridk.data[i].z;

        Fx = h_mesh_fft_Fx.data[i];
        Fy = h_mesh_fft_Fy.data[i];
        Fz = h_mesh_fft_Fz.data[i];


        // UF coupling
        // U_{real} = ya * sfuf^2 * (\mathbf{F}_{real} - (\mathbf{F}_{real} \cdot \mathbf{k}) * \mathbf{k})
        // U_{imgn} = ya * sfuf^2 * (\mathbf{F}_{imgn} - (\mathbf{F}_{imgn} \cdot \mathbf{k}) * \mathbf{k})
        Scalar ya = h_ymob.data[i].x;
        Scalar sf_uf = h_sf.data[i].x;
        Scalar uf_fac = ya * sf_uf * sf_uf;
        kiss_fft_cpx F_dot_k;
        F_dot_k.r = float(Fx.r * kx + Fy.r * ky + Fz.r * kz);
        F_dot_k.i = float(Fx.i * kx + Fy.i * ky + Fz.i * kz);

        h_mesh_fft_Fx.data[i].r = float( uf_fac * (Fx.r - F_dot_k.r * kx) );
        h_mesh_fft_Fx.data[i].i = float( uf_fac * (Fx.i - F_dot_k.i * kx) );

        h_mesh_fft_Fy.data[i].r = float( uf_fac * (Fy.r - F_dot_k.r * ky) );
        h_mesh_fft_Fy.data[i].i = float( uf_fac * (Fy.i - F_dot_k.i * ky) );

        h_mesh_fft_Fz.data[i].r = float( uf_fac * (Fz.r - F_dot_k.r * kz) );
        h_mesh_fft_Fz.data[i].i = float( uf_fac * (Fz.i - F_dot_k.i * kz) );
        } // for i
    
    } /* end of meshGreen */

/*! Conduct backward FFT of the velocity moments of mesh
 */
void TwoStepRPY::backwardFFT()
    {
    if (m_kiss_fft_initialized)
        {
        // backward FFT of mesh velocity
        // Ux
        ArrayHandle<kiss_fft_cpx> h_mesh_fft_Fx(m_mesh_fft_Fx,
                                                access_location::host, 
                                                access_mode::read);
        ArrayHandle<kiss_fft_cpx> h_mesh_inv_Fx(m_mesh_inv_Fx,
                                                access_location::host, 
                                                access_mode::overwrite);
        kiss_fftnd(m_kiss_ifft, 
                   h_mesh_fft_Fx.data, 
                   h_mesh_inv_Fx.data);
        // Uy
        ArrayHandle<kiss_fft_cpx> h_mesh_fft_Fy(m_mesh_fft_Fy,
                                                access_location::host, 
                                                access_mode::read);
        ArrayHandle<kiss_fft_cpx> h_mesh_inv_Fy(m_mesh_inv_Fy,
                                                access_location::host, 
                                                access_mode::overwrite); 
        kiss_fftnd(m_kiss_ifft, 
                   h_mesh_fft_Fy.data, 
                   h_mesh_inv_Fy.data);
        // Uz
        ArrayHandle<kiss_fft_cpx> h_mesh_fft_Fz(m_mesh_fft_Fz,
                                                access_location::host, 
                                                access_mode::read);
        ArrayHandle<kiss_fft_cpx> h_mesh_inv_Fz(m_mesh_inv_Fz,
                                                access_location::host, 
                                                access_mode::overwrite);
        kiss_fftnd(m_kiss_ifft, 
                   h_mesh_fft_Fz.data, 
                   h_mesh_inv_Fz.data);
        } // if m_kiss_fft_initialized

#ifdef ENABLE_MPI
    if (m_pdata->getDomainDecomposition())
        {
        // Distributed backward FFT of mesh velocity
        m_exec_conf->msg->notice(8) << "RPY: Distributed iFFT" << std::endl;
        // Ux
        ArrayHandle<kiss_fft_cpx> h_mesh_fft_Fx(m_mesh_fft_Fx,
                                                access_location::host, 
                                                access_mode::read);
        ArrayHandle<kiss_fft_cpx> h_mesh_inv_Fx(m_mesh_inv_Fx,
                                                access_location::host, 
                                                access_mode::overwrite);
        dfft_execute((cpx_t*)h_mesh_fft_Fx.data,
                     (cpx_t*)(h_mesh_inv_Fx.data + m_ghost_offset),
                     1,
                     m_dfft_plan_inverse);
        // Uy
        ArrayHandle<kiss_fft_cpx> h_mesh_fft_Fy(m_mesh_fft_Fy,
                                                access_location::host, 
                                                access_mode::read);
        ArrayHandle<kiss_fft_cpx> h_mesh_inv_Fy(m_mesh_inv_Fy,
                                                access_location::host, 
                                                access_mode::overwrite);
        dfft_execute((cpx_t*)h_mesh_fft_Fy.data,
                     (cpx_t*)(h_mesh_inv_Fy.data + m_ghost_offset),
                     1,
                     m_dfft_plan_inverse);                                                
        // Uz
        ArrayHandle<kiss_fft_cpx> h_mesh_fft_Fz(m_mesh_fft_Fz,
                                                access_location::host, 
                                                access_mode::read);
        ArrayHandle<kiss_fft_cpx> h_mesh_inv_Fz(m_mesh_inv_Fz,
                                                access_location::host, 
                                                access_mode::overwrite);
        dfft_execute((cpx_t*)h_mesh_fft_Fz.data,
                     (cpx_t*)(h_mesh_inv_Fz.data + m_ghost_offset),
                     1,
                     m_dfft_plan_inverse);
        }
#endif

#ifdef ENABLE_MPI
    if (m_pdata->getDomainDecomposition())
        {
        // update outer cells of force mesh using ghost cells from neighboring processors
        m_exec_conf->msg->notice(8) << "RPY: Ghost cell update" << std::endl;
        m_grid_comm_reverse->communicate(m_mesh_inv_Fx);
        m_grid_comm_reverse->communicate(m_mesh_inv_Fy);
        m_grid_comm_reverse->communicate(m_mesh_inv_Fz);
        }
#endif     
    
    } /* end of backwardFFT */

/*! Interpolate velocity moments of mesh back to particles

    uwave       output  wave-space velocity of particles
*/
void TwoStepRPY::interpolateParticleVelocity(Scalar * uwave)
    {
    unsigned int group_size = m_group->getNumMembers();
    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(),
                               access_location::host,
                               access_mode::read);
    const BoxDim& box = m_pdata->getBox();

    // access mesh data
    ArrayHandle<kiss_fft_cpx> h_mesh_inv_Fx(m_mesh_inv_Fx,
                                            access_location::host,
                                            access_mode::read);
    ArrayHandle<kiss_fft_cpx> h_mesh_inv_Fy(m_mesh_inv_Fy,
                                            access_location::host,
                                            access_mode::read);
    ArrayHandle<kiss_fft_cpx> h_mesh_inv_Fz(m_mesh_inv_Fz,
                                            access_location::host,
                                            access_mode::read);

    // loop over particles
    for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
        {
        unsigned int idx = m_group->getMemberIndex(group_idx);
        Scalar3 pos = make_scalar3(h_pos.data[idx].x, h_pos.data[idx].y, h_pos.data[idx].z);

        unsigned int i3 = idx * 3;
        uwave[i3    ] = 0.0;
        uwave[i3 + 1] = 0.0;
        uwave[i3 + 2] = 0.0;
        
        Scalar3 fpos = box.makeFraction(pos);
        Scalar3 reduced_pos = make_scalar3(fpos.x * (Scalar)m_mesh_points.x, 
                                           fpos.y * (Scalar)m_mesh_points.y,
                                           fpos.z * (Scalar)m_mesh_points.z);

        reduced_pos.x += (Scalar)m_n_ghost_cells.x;
        reduced_pos.y += (Scalar)m_n_ghost_cells.y;
        reduced_pos.z += (Scalar)m_n_ghost_cells.z;

        Scalar shift, shiftone;

        if (m_P % 2)
            {
            shift = 0.5;
            shiftone = 0.0;
            }
        else
            {
            shift = 0.0;
            shiftone = 0.5;
            }

        int ix = int(reduced_pos.x + shift);
        int iy = int(reduced_pos.y + shift);
        int iz = int(reduced_pos.z + shift);

        Scalar dx = shiftone + (Scalar)ix - reduced_pos.x;
        Scalar dy = shiftone + (Scalar)iy - reduced_pos.y;
        Scalar dz = shiftone + (Scalar)iz - reduced_pos.z;

        // handle particles on the boundary
        if (ix == (int)m_grid_dim.x && !m_n_ghost_cells.x)
            ix = 0;
        if (iy == (int)m_grid_dim.y && !m_n_ghost_cells.y)
            iy = 0;
        if (iz == (int)m_grid_dim.z && !m_n_ghost_cells.z)
            iz = 0;

        if (ix < 0 || ix >= (int)m_grid_dim.x || iy < 0 || iy >= (int)m_grid_dim.y || iz < 0
            || iz >= (int)m_grid_dim.z)
            {
            // ignore, error will be thrown elsewhere (in CellList)
            continue;
            }

        int nlower = -(static_cast<int>(m_P) - 1) / 2;
        int nupper = m_P / 2;

        for (int i = nlower; i <= nupper; ++i)
            {
            int neighi = (int)ix + i;
            Scalar rx = (Scalar(i) + dx) * m_h.x;

            if (!m_n_ghost_cells.x)
                {
                if (neighi >= (int)m_grid_dim.x)
                    neighi -= m_grid_dim.x;
                else if (neighi < 0)
                    neighi += m_grid_dim.x;
                }

            for (int j = nlower; j <= nupper; ++j)
                {
                int neighj = (int)iy + j;
                Scalar ry = (Scalar(j) + dy) * m_h.y;

                if (!m_n_ghost_cells.y)
                    {
                    if (neighj >= (int)m_grid_dim.y)
                        neighj -= m_grid_dim.y;
                    else if (neighj < 0)
                        neighj += m_grid_dim.y;
                    }
                
                for (int k = nlower; k <= nupper; ++k)
                    {
                    int neighk = (int)iz + k;
                    Scalar rz = (Scalar(k) + dz) * m_h.z;
                    
                    if (!m_n_ghost_cells.z)
                        {
                        if (neighk >= (int)m_grid_dim.z)
                            neighk -= m_grid_dim.z;
                        else if (neighk < 0)
                            neighk += m_grid_dim.z;
                        }

                    // store in row major order
                    unsigned int neigh_idx = neighi + m_grid_dim.x * (neighj + m_grid_dim.y * neighk);
                    Scalar r2 = rx * rx + ry * ry + rz * rz;
                    Scalar fac = m_vk * m_gauss_fac * fast::exp( - m_gauss_exp * r2 );

                    Scalar Ux = Scalar(h_mesh_inv_Fx.data[neigh_idx].r);
                    Scalar Uy = Scalar(h_mesh_inv_Fy.data[neigh_idx].r);
                    Scalar Uz = Scalar(h_mesh_inv_Fz.data[neigh_idx].r);

                    uwave[i3    ] += fac * Ux;
                    uwave[i3 + 1] += fac * Uy;
                    uwave[i3 + 2] += fac * Uz;
                    } // for k
                
                } // for j
            
            } // for i

        } // for group_idx

    } /* end of interpolateParticleVelocity*/

/*! Wrap up the calculation of the matrix-vector multiplication M . F in wave space
    where F may contain force, torque, and stresslet

    fts         input   force/torque/stresslet
    uwave       output  wave-space velocity/rotational velocity/rate of strain
 */
void TwoStepRPY::mobilityWaveUF(Scalar * force,
                                Scalar * uwave)
    {
    // assign force moments to mesh
    assignParticleForce(force);

    // //~ output data
    // printf("\nAfter force spreading\n");
    // ArrayHandle<kiss_fft_cpx> h_mesh_Fx(m_mesh_Fx,
    //                                      access_location::host,
    //                                      access_mode::read);
    // ArrayHandle<kiss_fft_cpx> h_mesh_Fy(m_mesh_Fy,
    //                                      access_location::host,
    //                                      access_mode::read);
    // ArrayHandle<kiss_fft_cpx> h_mesh_Fz(m_mesh_Fz,
    //                                      access_location::host,
    //                                      access_mode::read);
    // std::ofstream file("cpu_fft.dat");  
    // for (unsigned int i = 0; i < m_n_inner_cells; i++)
    //     {
    //     file << h_mesh_Fx.data[i].r << " " << h_mesh_Fy.data[i].r << " " << h_mesh_Fz.data[i].r << std::endl;
    //     } 
    // file.close();                         
    // // for (unsigned int i = 0; i < m_n_inner_cells; i++)
    // //     {
    // //     printf("Grid %u: (%f %f), (%f %f), (%f %f)\n", i, h_mesh_Fx.data[i].r, h_mesh_Fx.data[i].i, h_mesh_Fy.data[i].r, h_mesh_Fy.data[i].i, h_mesh_Fz.data[i].r, h_mesh_Fz.data[i].i);
    // //     }
    // //~

    // forward FFT
    forwardFFT();

    // //~ output data
    // printf("\nAfter FFT\n");
    // ArrayHandle<kiss_fft_cpx> h_mesh_fft_Fx(m_mesh_fft_Fx,
    //                                          access_location::host,
    //                                          access_mode::read);
    // ArrayHandle<kiss_fft_cpx> h_mesh_fft_Fy(m_mesh_fft_Fy,
    //                                          access_location::host,
    //                                          access_mode::read);
    // ArrayHandle<kiss_fft_cpx> h_mesh_fft_Fz(m_mesh_fft_Fz,
    //                                          access_location::host,
    //                                          access_mode::read);
    // for (unsigned int i = 0; i < m_n_inner_cells; i++)
    //     {
    //     printf("Grid %u: (%f %f), (%f %f), (%f %f)\n", i, h_mesh_fft_Fx.data[i].r, h_mesh_fft_Fx.data[i].i, h_mesh_fft_Fy.data[i].r, h_mesh_fft_Fy.data[i].i, h_mesh_fft_Fz.data[i].r, h_mesh_fft_Fz.data[i].i);
    //     }
    // //~

    // scale by Green's functions
    meshGreen();

    // //~ output data
    // printf("\nAfter scaling\n");
    // for (unsigned int i = 0; i < m_n_inner_cells; i++)
    //     {
    //     printf("Grid %u: (%f %f), (%f %f), (%f %f)\n", i, h_mesh_fft_Fx.data[i].r, h_mesh_fft_Fx.data[i].i, h_mesh_fft_Fy.data[i].r, h_mesh_fft_Fy.data[i].i, h_mesh_fft_Fz.data[i].r, h_mesh_fft_Fz.data[i].i);
    //     }
    // //~

    // backward FFT
    backwardFFT();

    // //~ output data
    // printf("\nAfter IFFT\n");
    // ArrayHandle<kiss_fft_cpx> h_mesh_inv_Fx(m_mesh_inv_Fx,
    //                                          access_location::host,
    //                                          access_mode::read);
    // ArrayHandle<kiss_fft_cpx> h_mesh_inv_Fy(m_mesh_inv_Fy,
    //                                          access_location::host,
    //                                          access_mode::read);
    // ArrayHandle<kiss_fft_cpx> h_mesh_inv_Fz(m_mesh_inv_Fz,
    //                                          access_location::host,
    //                                          access_mode::read);
    // for (unsigned int i = 0; i < m_n_inner_cells; i++)
    //     {
    //     printf("Grid %u: (%f %f), (%f %f), (%f %f)\n", i, h_mesh_inv_Fx.data[i].r, h_mesh_inv_Fx.data[i].i, h_mesh_inv_Fy.data[i].r, h_mesh_inv_Fy.data[i].i, h_mesh_inv_Fz.data[i].r, h_mesh_inv_Fz.data[i].i);
    //     }
    // //~

    // interpolate back to particle
    interpolateParticleVelocity(uwave);
    }

/*! Calculate determinant velocity U = M \cdot F
    where U is a generalized velocity that may also contain rotational velocity and rate of strain, F is a generalized force that may also contain torque and stresslet.
    The calculation is decomposed into a real-part Ur = Mr \cdot F and a wave-part Uw = Mw \cdot F, and then add them together: U = Ur + Uw
    
    fts     input   force/torque/stresslet 
    uoe     output  velocity/rotational velocity/rate of strain
 */ 
void TwoStepRPY::mobilityGeneralUF(Scalar * force,
                                   Scalar * velocity)
    {
    unsigned int group_size = m_group->getNumMembers();
    unsigned int numel = group_size * 3;
    // real-space calculation
    ArrayHandle<Scalar> h_u_real(m_u_real,
                                 access_location::host,
                                 access_mode::readwrite);
    mobilityRealUF(force,
                   h_u_real.data);

    //~ output data
    ArrayHandle<unsigned int> h_index_array(m_group->getIndexArray(),
                                            access_location::host,
                                            access_mode::read);                         
    printf("Real-part deterministic velocity\n");
    for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
        {
        unsigned int idx = h_index_array.data[group_idx];
        unsigned int i3 = idx * 3;
        printf("Particle %u: %f %f %f\n", idx, h_u_real.data[i3], h_u_real.data[i3 + 1], h_u_real.data[i3 + 2]);
        }
    //~

    // wave-space calculation
    ArrayHandle<Scalar> h_u_wave(m_u_wave,
                                 access_location::host,
                                 access_mode::readwrite);
    mobilityWaveUF(force,
                   h_u_wave.data);

    //~ output data
    printf("Wave-part deterministic velocity\n");
    for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
        {
        unsigned int idx = h_index_array.data[group_idx];
        unsigned int i3 = idx * 3;
        printf("Particle %u: %f %f %f\n", idx, h_u_wave.data[i3], h_u_wave.data[i3 + 1], h_u_wave.data[i3 + 2]);
        }
    //~

    // add up velocity
    for (unsigned int i = 0; i < numel; i++)
        {
        velocity[i] = h_u_real.data[i] + h_u_wave.data[i];
        } // for i

    //~ output data
    printf("Total deterministic velocity\n");
    for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
        {
        unsigned int idx = h_index_array.data[group_idx];
        unsigned int i3 = idx * 3;
        printf("Particle %u: %f %f %f\n", idx, velocity[i3], velocity[i3 + 1], velocity[i3 + 2]);
        }
    //~
    
    } /* end of mobilityGeneralUF */

/*! Generates random vector with 0 mean and sqrt(6T/dt)

    timestep    input   current time step that is used for generating random vector Psi in real-space
    psi         output  random vector     
 */
void TwoStepRPY::brownianFarFieldRealRNG(uint64_t timestep,
                                         Scalar * psi)
    {
    Scalar currentTemp = m_T->operator()(timestep);
    Scalar fac = fast::sqrt(6.0 * currentTemp / m_deltaT);

    unsigned int group_size = m_group->getNumMembers();
    uint16_t m_seed = m_sysdef->getSeed();

    ArrayHandle<unsigned int> h_tag(m_pdata->getTags(),
                                    access_location::host,
                                    access_mode::read);
    for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
        {
        unsigned int idx = m_group->getMemberIndex(group_idx);
        unsigned int ptag = h_tag.data[idx];

        unsigned int i3 = idx * 3;

        RandomGenerator rng(hoomd::Seed(RNGIdentifier::StokesMreal, timestep, m_seed), hoomd::Counter(ptag));
        UniformDistribution<Scalar> uniform(Scalar(-1), Scalar(1));

        // force
        psi[i3    ] = fac * uniform(rng);
        psi[i3 + 1] = fac * uniform(rng);
        psi[i3 + 2] = fac * uniform(rng);
        } // for group_idx
    
    } /* end of brownianFarFieldRealRNG */

/*! Calculates x = Tm . V

    V           input   basis vectors with size [n * m]
    Tm          input   multiplying vector with size [m]
    m           input   size of Tm
    n           input   number of rows of V
    x           ouput   vector with size [n * 1]
*/
inline void lanczosMapBack(Scalar * V,
                           double * Tm,
                           unsigned int m,
                           unsigned int numel,
                           Scalar * x)
    {
    for (unsigned int i = 0; i < numel; i++)
        {
        x[i] = 0.0;

        for (unsigned int j = 0; j < m; j++)
            {
            
            x[i] += V[j * numel + i] * Scalar(Tm[j]);
            
            } // for j
        
        } // for i

    } /* end of lanczosTriSqrt */

/*! Calculates M^0.5 \cdot Psi using Lanczos algorithm
    
    psi             input       random vector
    iter_ff_v       input       vector for Lanczos iteration
    iter_ff_vj      input       vector for Lanczos iteration
    iter_ff_vjm1    input       vector for Lanczos iteration
    iter_ff_V       input       basis vectors for Lanczos iteration
    iter_ff_uold    input       old value of Brownian slip velocity
    u               output      Brownian slip velocity
 */
void TwoStepRPY::brownianLanczos(Scalar * psi,
                                 Scalar * iter_ff_v,
                                 Scalar * iter_ff_vj,
                                 Scalar * iter_ff_vjm1,
                                 Scalar * iter_ff_V,
                                 Scalar * iter_ff_uold,
                                 Scalar * iter_Mpsi,
                                 Scalar * u)
    {
    unsigned int group_size = m_group->getNumMembers();
    unsigned int numel = group_size * 3;

    unsigned int m_in = m_m_lanczos_ff;
    unsigned int mmax = m_mmax;

    Scalar * v = iter_ff_v;
    Scalar * V = iter_ff_V;
    Scalar * vj = iter_ff_vj;
    Scalar * vjm1 = iter_ff_vjm1;
    Scalar * uold = iter_ff_uold;
    Scalar * Mpsi = iter_Mpsi;
    Scalar * temp;

    double * alpha = new double[mmax];
    double * alpha_save = new double[mmax];
    double * beta = new double[mmax + 1];
    double * beta_save = new double[mmax + 1];
    double * W = new double[mmax * mmax];
    double * W1 = new double[mmax];
    double * Tm = new double[mmax];

    // copy psi to v0
    memcpy(vj, psi, numel * sizeof(Scalar));

    Scalar vnorm, psinorm;
    vnorm = cblas_dnrm2(numel, vj, 1);
    psinorm = vnorm;

    // compute psi . M . psi (for step norm)
    mobilityRealUF(psi,
                   Mpsi);
    Scalar psiMpsi = cblas_ddot(numel,
                                psi, 1,
                                Mpsi, 1);
    psiMpsi = psiMpsi / (psinorm * psinorm);

    unsigned int m = m_in - 1;
    m = m < 1 ? 1 : m;

    Scalar alpha_temp;
    Scalar beta_temp = 0.0;

    // first iteration
    // scale v_{j-1} by 0 and v_{j} by 1 / vnorm
    Scalar scale = 0.0;
    cblas_dscal(numel,
                scale,
                vjm1, 1);
    scale = 1.0 / vnorm;
    cblas_dscal(numel,
                scale,
                vj, 1);
    
    for (unsigned int j = 0; j < m; j++)
        {
        memcpy(&V[j * numel], vj, numel * sizeof(Scalar));
        beta[j] = beta_temp;

        // v = M . v_{j} - beta_{j} * v_{j - 1}
        mobilityRealUF(vj,
                       v);
        scale = - beta_temp;
        cblas_daxpy(numel,
                    scale,
                    vjm1, 1,
                    v, 1);

        // alpha_{j} = v_{j} \cdot v
        alpha_temp = cblas_ddot(numel,
                                vj, 1,
                                v, 1);
        alpha[j] = alpha_temp;
        
        // v = v - alpha_{j} * v_{j}
        scale = - alpha_temp;
        cblas_daxpy(numel,
                    scale,
                    vj, 1,
                    v, 1);

        // beta_{j + 1} = v \cdot v
        beta_temp = cblas_dnrm2(numel, v, 1);

        if (beta_temp < 1e-8)
            {
            m = j + 1;
            break;
            }
        
        // v = v / beta_{j + 1} 
        scale = 1.0 / beta_temp;
        cblas_dscal(numel,
                    scale,
                    v, 1);

        // store current basis vector V
        for (unsigned int i = 0; i < numel; i++)
            {
            V[(j + 2) * numel + i] = v[i];
            }
        
        // swap pointers
        temp = vjm1;
        vjm1 = vj;
        vj = v;
        v = temp;
        } // for j

    // save alpha and beta (will be overwritten by LAPACK)
    for (unsigned int i = 0; i < m; i++)
        {
        alpha_save[i] = alpha[i];
        beta_save[i] = beta[i];
        }
    beta_save[m] = beta[m];

    // compute eigen-decomposition of tridiagonal matrix
    int INFO = LAPACKE_dpteqr(LAPACK_ROW_MAJOR, 'I', m, alpha, &beta[1], W, m);
    if (INFO != 0)
        {
        std::cout << "Eigenvalue decomposition #1 failed." << std::endl;
        std::cout << "INFO = " << INFO << std::endl;
        std::cout << "m = " << m << std::endl;
            std::cout << "alpha: " << std::endl;
            for (unsigned int k = 0; k < m; k++)
                {
                std::cout << alpha[k] << std::endl;
                }
            std::cout << std::endl;

            std::cout << "beta: " << std::endl;
            for (unsigned int k = 0; k < m; k++)
                {
                std::cout << beta[k] << std::endl;
                }
            std::cout << std::endl;
        exit(EXIT_FAILURE); 
        }
    for (unsigned int i = 0; i < m; i++)
        {
        W1[i] = sqrt(alpha[i]) * W[i];
        }
    double sum_temp;
    for (unsigned int i = 0; i < m; i++)
        {
        sum_temp = 0.0;
        for (unsigned int k = 0; k < m; k++)
            {
            unsigned int idx = i * m + k;
            sum_temp += W[idx] * W1[k];
            } // for j

            Tm[i] = sum_temp;
        }

    // multiply basis vectors by Tm to get x = A \cdot Tm
    lanczosMapBack(V,
                   Tm,
                   m,
                   numel,
                   u);
    memcpy(uold, u, numel * sizeof(Scalar));

    // recover alpha and beta
    for (unsigned int i = 0; i < m; i++)
        {
        alpha[i] = alpha_save[i];
        beta[i] = beta_save[i];
        }
    beta[m] = beta_save[m];
    
    // std::cout << "Initial Lanczos iteration is done" << std::endl;
    // keep adding to basis vectors untill step norm is small enough
    Scalar stepnorm = 1.0;
    unsigned int j;
    while (stepnorm > m_error && m < mmax)
        {
        m++;
        j = m - 1;

        memcpy(&V[j * numel], vj, numel * sizeof(Scalar));
        beta[j] = beta_temp;

        // v = M . v_{j} - beta_{j} * v_{j - 1}
        mobilityRealUF(vj,
                       v);
        scale = - beta_temp;
        cblas_daxpy(numel,
                    scale,
                    vjm1, 1,
                    v, 1);

        // alpha_{j} = v_{j} . v
        alpha_temp = cblas_ddot(numel,
                                vj, 1,
                                v, 1);
        // store alpha_{j}
        alpha[j] = alpha_temp;
        
        // v = v - alpha_{j} * v_{j}
        scale = - alpha_temp;
        cblas_daxpy(numel,
                    scale,
                    vj, 1,
                    v, 1);

        // beta_{j + 1} = v \cdot v
        beta_temp = cblas_dnrm2(numel, v, 1);
        beta[j + 1] = beta_temp;
        
        if (beta_temp < 1e-8)
            {
            m = j + 1;
            break;
            }
        
        // v = v / beta_{j + 1} 
        scale = 1.0 / beta_temp;
        cblas_dscal(numel,
                    scale,
                    v, 1);

        temp = vjm1;
        vjm1 = vj;
        vj = v;
        v = temp;

        // save alpha and beta (will be overwritten by LAPACK)
        for (unsigned int i = 0; i < m; i++)
            {
            alpha_save[i] = alpha[i];
            beta_save[i] = beta[i];
            }
        beta_save[m] = beta[m];

        // compute eigen-decomposition of tridiagonal matrix
        int INFO = LAPACKE_dpteqr(LAPACK_ROW_MAJOR, 'I', m, alpha, &beta[1], W, m);
        if (INFO != 0)
            {
            std::cout << "Eigenvalue decomposition #2 failed." << std::endl;
            std::cout << "INFO = " << INFO << std::endl;
            std::cout << "m = " << m << std::endl;
            std::cout << "alpha: " << std::endl;
            for (unsigned int k = 0; k < m; k++)
                {
                std::cout << alpha[k] << std::endl;
                }
            std::cout << std::endl;

            std::cout << "beta: " << std::endl;
            for (unsigned int k = 0; k < m; k++)
                {
                std::cout << beta[k] << std::endl;
                }
            std::cout << std::endl;
            exit(EXIT_FAILURE); 
            }
        for (unsigned int i = 0; i < m; i++)
            {
            W1[i] = sqrt(alpha[i]) * W[i];
            }
        
        for (unsigned int i = 0; i < m; i++)
            {
            sum_temp = 0.0;
            for (unsigned int k = 0; k < m; k++)
                {
                unsigned int idx = i * m + k;
                sum_temp += W[idx] * W1[k];
                } // for j

                Tm[i] = sum_temp;
            }
        // multiply basis vectors by Tm to get x = A \cdot Tm
        lanczosMapBack(V,
                       Tm,
                       m,
                       numel,
                       u);

        // compute step norm
        scale = - 1.0;
        cblas_daxpy(numel,
                    scale,
                    u, 1,
                    uold, 1);
        stepnorm = cblas_ddot(numel,
                              uold, 1,
                              uold, 1);
        stepnorm = sqrt(stepnorm / psiMpsi);
        memcpy(uold, u, numel * sizeof(Scalar));

        // recover alpha and beta
        for (unsigned int i = 0; i < m; i++)
            {
            alpha[i] = alpha_save[i];
            beta[i] = beta_save[i];
            }
            beta[m] = beta_save[m];
        } // while

    // std::cout << "Lanczos iteration number = " << m << std::endl;

    // scale by the norm of psi
    cblas_dscal(numel,
                psinorm,
                u, 1); 
    
    v = NULL;
    vj = NULL;
    vjm1 = NULL;
    V = NULL;
    uold = NULL;
    Mpsi = NULL;
    temp = NULL;

    delete [] alpha;
    delete [] alpha_save;
    delete [] beta;
    delete [] beta_save;
    delete [] W;
    delete [] W1;
    delete [] Tm;
    } /* end of brownianLanczos */

/*! Wraps up the calculation of the real-space Brownian far-field slip velocity

    timestep    input   current time step that is used for generating random vector Psi in real-space
    ureal       output  far-field Brownian slip velocity in real space    
 */
void TwoStepRPY::brownianFarFieldSlipVelocityReal(uint64_t timestep,
                                                  Scalar * ureal)
    {
    // std::cout << "Brownian Real" << std::endl;
    // generate random vector \Psi 
    ArrayHandle<Scalar> h_iter_psi(m_iter_psi,
                                   access_location::host,
                                   access_mode::readwrite);
    brownianFarFieldRealRNG(timestep,
                            h_iter_psi.data);

    // solve for M_{real}^{0.5} \cdot \Psi_{real} by using Lanczos iterations
    ArrayHandle<Scalar> h_iter_ff_v(m_iter_ff_v,
                                    access_location::host,
                                    access_mode::readwrite);
    ArrayHandle<Scalar> h_iter_ff_vj(m_iter_ff_vj,
                                     access_location::host,
                                     access_mode::readwrite);
    ArrayHandle<Scalar> h_iter_ff_vjm1(m_iter_ff_vjm1,
                                       access_location::host,
                                       access_mode::readwrite);
    ArrayHandle<Scalar> h_iter_ff_V(m_iter_ff_V,
                                    access_location::host,
                                    access_mode::readwrite);
    ArrayHandle<Scalar> h_iter_ff_uold(m_iter_ff_uold,
                                       access_location::host,
                                       access_mode::readwrite);
    ArrayHandle<Scalar> h_iter_Mpsi(m_iter_Mpsi,
                                    access_location::host,
                                    access_mode::readwrite);
                                    
    brownianLanczos(h_iter_psi.data,
                    h_iter_ff_v.data,
                    h_iter_ff_vj.data,
                    h_iter_ff_vjm1.data,
                    h_iter_ff_V.data,
                    h_iter_ff_uold.data,
                    h_iter_Mpsi.data,
                    ureal);
    
    } /* end of brownianFarFieldSlipVelocityReal */

/*! Generates random vectors of mesh with proper conjugacy.
    Scale random vectors with square root of wave space contribution to the Ewald sum

    timestep    input   current time step that is used for generating random vector Psi
*/
void TwoStepRPY::brownianFarFieldGridRNG(uint64_t timestep)
    {
    Scalar currentTemp = m_T->operator()(timestep);
    Scalar fac = fast::sqrt(Scalar(3.0) * currentTemp / m_deltaT / m_vk);
    Scalar sqrt2 = sqrt(2.0);

    uint16_t m_seed = m_sysdef->getSeed();
    // get random vector of global mesh with proper conjugacy
    ArrayHandle<kiss_fft_cpx> h_mesh_fft_Fx(m_mesh_fft_Fx,
                                            access_location::host,
                                            access_mode::overwrite);
    ArrayHandle<kiss_fft_cpx> h_mesh_fft_Fy(m_mesh_fft_Fy,
                                            access_location::host,
                                            access_mode::overwrite);
    ArrayHandle<kiss_fft_cpx> h_mesh_fft_Fz(m_mesh_fft_Fz,
                                            access_location::host,
                                            access_mode::overwrite);

    // zero mesh
    memset(h_mesh_fft_Fx.data, 0, sizeof(kiss_fft_cpx) * m_mesh_fft_Fx.getNumElements());
    memset(h_mesh_fft_Fy.data, 0, sizeof(kiss_fft_cpx) * m_mesh_fft_Fy.getNumElements());
    memset(h_mesh_fft_Fz.data, 0, sizeof(kiss_fft_cpx) * m_mesh_fft_Fz.getNumElements());

    unsigned int nkx = m_global_dim.x;
    unsigned int nky = m_global_dim.y;
    unsigned int nkz = m_global_dim.z;

    // step 1: generate random vector with conjugacy on global mesh
    for (unsigned int cell_idx = 0; cell_idx < m_NNN; cell_idx++)
        {
        RandomGenerator rng(hoomd::Seed(RNGIdentifier::StokesMwave, timestep, m_seed), hoomd::Counter(cell_idx));
        UniformDistribution<Scalar> uniform(Scalar(-1), Scalar(1));        

        Scalar re_x = fac * uniform(rng);
        Scalar re_y = fac * uniform(rng);
        Scalar re_z = fac * uniform(rng);

        Scalar im_x = fac * uniform(rng);
        Scalar im_y = fac * uniform(rng);
        Scalar im_z = fac * uniform(rng);

        unsigned int k = cell_idx / nky / nkx;
        unsigned int j = (cell_idx - k * nky * nkx) / nkx;
        unsigned int i = cell_idx % nkx;

        // only do half the grid
        if ( !(2 * k >= nkz + 1) &&     // lower half of the cube across the z-plane
             !( (k == 0) && (2 * j >= nky + 1) ) &&     // lower half of the plane across the y-line
             !( (k == 0) && (j == 0) && (2 * i >= nkx + 1) ) &&     // lower half of the line across the x-point
             !( (k == 0) && (j == 0) && (i == 0) )      // ignore origin
           ) 
            {
            bool i_nyquist = ( (i == nkx / 2) && (nkx / 2 == (nkx + 1) / 2) );
            bool j_nyquist = ( (j == nky / 2) && (nky / 2 == (nky + 1) / 2) );
            bool k_nyquist = ( (k == nkz / 2) && (nkz / 2 == (nkz + 1) / 2) );

            // index of conjugate point
            unsigned int i_conj, j_conj, k_conj;
            if (i == 0 || i_nyquist)
                {
                i_conj = i;
                }
            else
                {
                i_conj = nkx - i;
                }
            
            if (j == 0 || j_nyquist)
                {
                j_conj = j;
                }
            else
                {
                j_conj = nky - j;
                }

            if (k == 0 || k_nyquist)
                {
                k_conj = k;
                }
            else
                {
                k_conj = nkz - k;
                }
            
            unsigned int id_conj = i_conj + nkx * (j_conj + nky * k_conj);
            if (cell_idx > id_conj)
                {
                continue;
                }

            if ( (i == 0    &&  j_nyquist   && k == 0   ) ||
                 (i_nyquist &&  j == 0      && k == 0   ) ||
                 (i_nyquist &&  j_nyquist   && k == 0   ) ||
                 (i == 0    &&  j == 0      && k_nyquist) ||
                 (i == 0    &&  j_nyquist   && k_nyquist) ||
                 (i_nyquist &&  j == 0      && k_nyquist) ||
                 (i_nyquist &&  j_nyquist   && k_nyquist)
               )
                {
                // only real part exsists. need to rescale by 2^0.5 to have variance 1
                h_mesh_fft_Fx.data[cell_idx].r = float(sqrt2 * re_x);
                h_mesh_fft_Fx.data[cell_idx].i = 0.0;
                h_mesh_fft_Fy.data[cell_idx].r = float(sqrt2 * re_y);
                h_mesh_fft_Fy.data[cell_idx].i = 0.0;
                h_mesh_fft_Fz.data[cell_idx].r = float(sqrt2 * re_z);
                h_mesh_fft_Fz.data[cell_idx].i = 0.0;
                }
            else
                {
                h_mesh_fft_Fx.data[cell_idx].r = float(re_x);
                h_mesh_fft_Fx.data[cell_idx].i = float(im_x);

                h_mesh_fft_Fy.data[cell_idx].r = float(re_y);
                h_mesh_fft_Fy.data[cell_idx].i = float(im_y);

                h_mesh_fft_Fz.data[cell_idx].r = float(re_z);
                h_mesh_fft_Fz.data[cell_idx].i = float(im_z);

                // conjugate points
                h_mesh_fft_Fx.data[id_conj].r = + float(re_x);
                h_mesh_fft_Fx.data[id_conj].i = - float(im_x);

                h_mesh_fft_Fy.data[id_conj].r = + float(re_y);
                h_mesh_fft_Fy.data[id_conj].i = - float(im_y);

                h_mesh_fft_Fz.data[id_conj].r = + float(re_z);
                h_mesh_fft_Fz.data[id_conj].i = - float(im_z);

                } // check for nyquist
            
            } // if (half the grid)

        } // for cell_idx

    // step 2 scale with square root of wave function and map to local mesh
    ArrayHandle<Scalar4> h_gridk(m_gridk,
                                 access_location::host,
                                 access_mode::read);

    ArrayHandle<Scalar3> h_ymob(m_ymob,
                                access_location::host,
                                access_mode::read);
    ArrayHandle<Scalar2> h_sf(m_sf,
                              access_location::host,
                              access_mode::read);

    for (unsigned int cell_idx = 0; cell_idx < m_n_inner_cells; cell_idx++)
        {
        uint3 wave_idx;

        // kiss FFT expects data in row major format
        wave_idx.z = cell_idx / (m_mesh_points.y * m_mesh_points.x);
        wave_idx.y
            = (cell_idx - wave_idx.z * m_mesh_points.x * m_mesh_points.y) / m_mesh_points.x;
        wave_idx.x = cell_idx % m_mesh_points.x;
        
        int3 n = make_int3(wave_idx.x, wave_idx.y, wave_idx.z);
        // compute Miller indices
        int cell_idx_global = n.x + m_global_dim.x * (n.y + m_global_dim.y * n.z);

        Scalar kx = h_gridk.data[cell_idx].x;
        Scalar ky = h_gridk.data[cell_idx].y;
        Scalar kz = h_gridk.data[cell_idx].z;
        // Scalar ks = h_gridk.data[cell_idx].w;
        
        Scalar B = (cell_idx == 0) ? 0.0 : sqrt(h_ymob.data[cell_idx].x);
        Scalar sfu = (cell_idx == 0) ? 0.0 : h_sf.data[cell_idx].x;
        Scalar sfc = (cell_idx == 0) ? 0.0 : h_sf.data[cell_idx].y;

        // conjugate
        sfc = - sfc;

        kiss_fft_cpx Fx = h_mesh_fft_Fx.data[cell_idx_global];
        kiss_fft_cpx Fy = h_mesh_fft_Fy.data[cell_idx_global];
        kiss_fft_cpx Fz = h_mesh_fft_Fz.data[cell_idx_global];

        // F . k
        kiss_fft_cpx F_dot_k;
        if (cell_idx == 0)
            {
            F_dot_k.r = 0.0;
            F_dot_k.i = 0.0;
            }
        else
            {
            F_dot_k.r = float(kx * Fx.r + ky * Fy.r + kz * Fz.r);
            F_dot_k.i = float(kx * Fx.i + ky * Fy.i + kz * Fz.i);
            }
        
        // B * (I - kk) . F
        kiss_fft_cpx BdWx, BdWy, BdWz;
        BdWx.r = float((Fx.r - kx * F_dot_k.r) * B);
        BdWx.i = float((Fx.i - kx * F_dot_k.i) * B);

        BdWy.r = float((Fy.r - ky * F_dot_k.r) * B);
        BdWy.i = float((Fy.i - ky * F_dot_k.i) * B);

        BdWz.r = float((Fz.r - kz * F_dot_k.r) * B);
        BdWz.i = float((Fz.i - kz * F_dot_k.i) * B);

        // velocity
        h_mesh_fft_Fx.data[cell_idx].r = float(sfu * BdWx.r);
        h_mesh_fft_Fx.data[cell_idx].i = float(sfu * BdWx.i);

        h_mesh_fft_Fy.data[cell_idx].r = float(sfu * BdWy.r);
        h_mesh_fft_Fy.data[cell_idx].i = float(sfu * BdWy.i);

        h_mesh_fft_Fz.data[cell_idx].r = float(sfu * BdWz.r);
        h_mesh_fft_Fz.data[cell_idx].i = float(sfu * BdWz.i);
        } // for cell_idx

    } /* end of brownianFarFieldGridRNG */

/*! Wraps up the calculation of the far-field Brownian slip velocity in wave space

    timestep    input   current time step that is used for generating random vector Psi
    uslip       output  far-field Brownian slip velocity of particles in wave space
*/
void TwoStepRPY::brownianFarFieldSlipVelocityWave(uint64_t timestep,
                                                  Scalar * uwave)
    {
    // std::cout << "Brownian Wave" << std::endl;
    // Generate random number of grid points with proper conjugacy
    // This step combines (1) Force distribution, (2) FFT, and (3) Scale by Green's function into one step to reduce FFT
    brownianFarFieldGridRNG(timestep);

    // backward FFT
    backwardFFT();

    // interpolate back to particle
    interpolateParticleVelocity(uwave);

    }

/*! Wraps up the calculation the far-field Brownian slip velocity, such that
    Uslip = M^0.5 \cdot \Psi
    in two parts, a real part and a wave part,
    Uslip_{real} = M_{real}^{0.5} \cdot \Psi_{real}
    Uslip_{wave} = M_{wave}^{0.5} \cdot \Psi_{wave}.
    where \Psi_{real} and \Psi_{wave} are random vectors generated independently

    timestep        input   current time step that is used for generating random vector Psi
    uslip           output  far-field Brownian slip velocity of particles
 */
void TwoStepRPY::brownianFarFieldSlipVelocity(uint64_t timestep,
                                              Scalar * uslip)
    {
    unsigned int group_size = m_group->getNumMembers();
    unsigned int numel = group_size * 3;

    // solve for the real part 
    ArrayHandle<Scalar> h_iter_ff_u(m_iter_ff_u,
                                    access_location::host,
                                    access_mode::readwrite);
    memset(h_iter_ff_u.data, 0, sizeof(Scalar) * m_iter_ff_u.getNumElements());
    brownianFarFieldSlipVelocityReal(timestep,
                                     h_iter_ff_u.data);

    // solve for the wave part
    ArrayHandle<Scalar> h_uslip_wave(m_uslip_wave,
                                     access_location::host,
                                     access_mode::readwrite);
    brownianFarFieldSlipVelocityWave(timestep,
                                     h_uslip_wave.data);
    
    for (unsigned int i = 0; i < numel; i++)
        {
        uslip[i] = h_iter_ff_u.data[i] + h_uslip_wave.data[i];
        }

    // //~ output data
    // ArrayHandle<unsigned int> h_index_array(m_group->getIndexArray(),
    //                                         access_location::host,
    //                                         access_mode::read);
    // printf("\nBrownain velocity\n");                                
    // printf("Real-part brownian velocity\n");
    // for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
    //     {
    //     unsigned int idx = h_index_array.data[group_idx];
    //     unsigned int i3 = idx * 3;
    //     printf("Particle %u: %f %f %f\n", idx, h_iter_ff_u.data[i3], h_iter_ff_u.data[i3 + 1], h_iter_ff_u.data[i3 + 2]);
    //     }
    // printf("Wave-part brownian velocity\n");
    // for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
    //     {
    //     unsigned int idx = h_index_array.data[group_idx];
    //     unsigned int i3 = idx * 3;
    //     printf("Particle %u: %f %f %f\n", idx, h_uslip_wave.data[i3], h_uslip_wave.data[i3 + 1], h_uslip_wave.data[i3 + 2]);
    //     }
    // printf("Total brownian velocity\n");
    // for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
    //     {
    //     unsigned int idx = h_index_array.data[group_idx];
    //     unsigned int i3 = idx * 3;
    //     printf("Particle %u: %f %f %f\n", idx, uslip[i3], uslip[i3 + 1], uslip[i3 + 2]);
    //     }
    // //~
    

    } /* end of brownianFarFieldSlipVelocity */

/*! Solve for velocity where dissipation is only due to far-field hydrodynamics
    U = M . F + M^0.5 . Psi
    Cannot be called if lubrication is on or dissipative contact mode is on.

    timestep    input   current timestep that is used for generating random vector Psi
    fts         input   conservative force/torque/stresslet of particles
    uoe         output  velocity of particles
 */
void TwoStepRPY::solverMobilityUF(uint64_t timestep,
                                  Scalar * force,
                                  Scalar * uoe)
    {
    // std::cout << "Mobility solver" << std::endl;
    unsigned int group_size = m_group->getNumMembers();
    unsigned int numel = group_size * 3;
    
    // calculate determinant velocity U = M . F
    ArrayHandle<Scalar> h_u_determin(m_u_determin,
                                     access_location::host,
                                     access_mode::readwrite);
    mobilityGeneralUF(force,
                      h_u_determin.data);
    
    // calculate Brownian velocity
    Scalar currentTemp = m_T->operator()(timestep);
    if (currentTemp > 0.0)
        {
        ArrayHandle<Scalar> h_uslip(m_uslip,
                                    access_location::host,
                                    access_mode::readwrite);
        brownianFarFieldSlipVelocity(timestep,
                                    h_uslip.data);

        // add Brownian velocity
        for (unsigned int i = 0; i < numel; i++)
            {
            uoe[i] = h_u_determin.data[i] + h_uslip.data[i];
            }
        
        } // if T > 0

    // //~ output data
    // printf("\nBefore adding up all velocity\n");
    // ArrayHandle<unsigned int> h_index_array(m_group->getIndexArray(),
    //                                         access_location::host,
    //                                         access_mode::read);

    // printf("Total deteriministic velocity\n");
    // for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
    //     {
    //     unsigned int idx = h_index_array.data[group_idx];
    //     unsigned int i3 = idx * 3;
    //     printf("Particle %u: %f %f %f\n", idx, h_u_determin.data[i3], h_u_determin.data[i3 + 1], h_u_determin.data[i3 + 2]);
    //     }

    // printf("Total brownian velocity\n");
    // ArrayHandle<Scalar> h_uslip(m_uslip,
    //                             access_location::host,
    //                             access_mode::read);                       
    // for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
    //     {
    //     unsigned int idx = h_index_array.data[group_idx];
    //     unsigned int i3 = idx * 3;
    //     printf("Particle %u: %f %f %f\n", idx, h_uslip.data[i3], h_uslip.data[i3 + 1], h_uslip.data[i3 + 2]);
    //     }

    // printf("Total velocity\n");
    // for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
    //     {
    //     unsigned int idx = h_index_array.data[group_idx];
    //     unsigned int i3 = idx * 3;
    //     printf("Particle %u: %f %f %f\n", idx, uoe[i3], uoe[i3 + 1], uoe[i3 + 2]);
    //     }
    // //~
    
    } /* end of solverMobilityUF */

void TwoStepRPY::mobilityMatrix()
    {
    unsigned int group_size = m_group->getNumMembers();
    unsigned int numel = group_size * 3;
    const BoxDim& box = m_pdata->getGlobalBox();
    Scalar3 L = box.getL();
    Scalar V_box = box.getVolume();
    Scalar pivol = M_PI / V_box;

    Scalar rcut = m_rcut_ewald;

    ArrayHandle<unsigned int> h_nneigh(m_nlist->getNNeighArray(),
                                       access_location::host,
                                       access_mode::read);
    ArrayHandle<unsigned int> h_nlist(m_nlist->getNListArray(),
                                      access_location::host,
                                      access_mode::read);
    ArrayHandle<size_t> h_headlist(m_nlist->getHeadList(),
                                   access_location::host,
                                   access_mode::read);

    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(),
                               access_location::host,
                               access_mode::read);

    ArrayHandle<Scalar> h_mobmat_real(m_mobmat_real,
                                      access_location::host,
                                      access_mode::overwrite);

    ArrayHandle<Scalar> h_mobmat_wave(m_mobmat_wave,
                                      access_location::host,
                                      access_mode::overwrite);   

    ArrayHandle<Scalar> h_kron(m_kron,
                               access_location::host,
                               access_mode::overwrite);                              
    memset(h_mobmat_real.data, 0, sizeof(Scalar) * numel * numel);
    memset(h_mobmat_wave.data, 0, sizeof(Scalar) * numel * numel);

    h_kron.data[0] = 1.0;   h_kron.data[1] = 0.0;   h_kron.data[2] = 0.0;
    h_kron.data[3] = 0.0;   h_kron.data[4] = 1.0;   h_kron.data[5] = 0.0;
    h_kron.data[6] = 0.0;   h_kron.data[7] = 0.0;   h_kron.data[8] = 1.0;

    // real-part mobility matrix
    for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
        {
        unsigned int idx = m_group->getMemberIndex(group_idx);
        unsigned int i3 = idx * 3;

        Scalar3 posi = make_scalar3(h_pos.data[idx].x, h_pos.data[idx].y, h_pos.data[idx].z);

        unsigned int nneigh_i = h_nneigh.data[idx];
        size_t head_i = h_headlist.data[idx];

        for (unsigned int neigh_idx = 0; neigh_idx < nneigh_i; neigh_idx++)
            {
            unsigned int jdx = h_nlist.data[head_i + neigh_idx];
            unsigned int j3 = jdx * 3;

            Scalar3 posj = make_scalar3(h_pos.data[jdx].x, h_pos.data[jdx].y, h_pos.data[jdx].z);
            Scalar3 dist = posj - posi;
            dist = box.minImage(dist);
            Scalar r2 = dot(dist, dist);
            Scalar r = fast::sqrt(r2);

            if (r < rcut)
                {
                Scalar3 e = dist / r;
                double d[3];
                d[0] = e.x;
                d[1] = e.y;
                d[2] = e.z;

                if (r < 2.0)
                    {
                    r = 2.0;
                    }

                Scalar xa, ya, yb, xc, yc;
                mobilityRealFunc(r,
                                 xa, ya,
                                 yb,
                                 xc, yc);

                for (unsigned int i = 0; i < 3; i++)
                    {
                    unsigned int row_ind = i3 + i;

                    for (unsigned int j = 0; j < 3; j++)
                        {
                        unsigned int col_ind = j3 + j;

                        h_mobmat_real.data[row_ind * numel + col_ind] = xa * d[i] * d[j] + ya * (h_kron.data[i * 3 + j] - d[i] * d[j]);
                        }
                    
                    }

                } // if r < rcut_ewald

            } // for neigh_idx

        } // for group_idx
    
    for (unsigned int i = 0; i < numel; i++)
    {
        h_mobmat_real.data[i * numel + i] = m_self_func.x;
    }

    // wave-part mobility matrix
    for (unsigned int cell_idx = 0; cell_idx < m_n_inner_cells; cell_idx++)
        {
        int mz = cell_idx / (m_mesh_points.x * m_mesh_points.y);
        int my = (cell_idx - mz * m_mesh_points.x * m_mesh_points.y) / m_mesh_points.x;
        int mx = cell_idx % m_mesh_points.x;

        if (mx >= (int(m_mesh_points.x) / 2 + int(m_mesh_points.x) % 2))
            {
            mx -= int(m_mesh_points.x);
            }

        if (my >= (int(m_mesh_points.y) / 2 + int(m_mesh_points.y) % 2))
            {
            my -= int(m_mesh_points.y);
            }

        if (mz >= (int(m_mesh_points.z) / 2 + int(m_mesh_points.z) % 2))
            {
            mz -= int(m_mesh_points.z);
            }

        if (cell_idx != 0)
            {
            Scalar kx = 2.0 * M_PI * Scalar(mx) / L.x;
            Scalar ky = 2.0 * M_PI * Scalar(my) / L.y;
            Scalar kz = 2.0 * M_PI * Scalar(mz) / L.z;
            Scalar k2 = kx * kx + ky * ky + kz * kz;
            Scalar k = sqrt(k2);

            double e[3];
            e[0] = kx / k;
            e[1] = ky / k;
            e[2] = kz / k;
            
            Scalar kxi = k2 / (4.0 * m_xi2);
            Scalar kexp = exp(- kxi);
            Scalar ya = 6.0 * pivol * (1.0 + kxi) / k2 * kexp;
            Scalar sf = 1.0 - k2 / 6.0;

            for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
                {
                unsigned int idx = m_group->getMemberIndex(group_idx);
                unsigned int i3 = idx * 3;

                Scalar3 posi = make_scalar3(h_pos.data[idx].x, h_pos.data[idx].y, h_pos.data[idx].z);

                for (unsigned int group_jdx = 0; group_jdx < group_size; group_jdx++)
                    {
                    unsigned int jdx = m_group->getMemberIndex(group_jdx);
                    unsigned int j3 = jdx * 3;

                    Scalar3 posj = make_scalar3(h_pos.data[jdx].x, h_pos.data[jdx].y, h_pos.data[jdx].z);

                    Scalar3 dist = posj - posi;
                    Scalar dx = dist.x;
                    Scalar dy = dist.y;
                    Scalar dz = dist.z;

                    Scalar cf = cos(kx * dx + ky * dy + kz * dz);
                    Scalar uf_fac = cf * ya * sf * sf;

                    for (unsigned int i = 0; i < 3; i++)
                        {
                        unsigned int row_idx = i3 + i;

                        for (unsigned int j = 0; j < 3; j++)
                            {
                            unsigned int col_idx = j3 + j;

                            h_mobmat_wave.data[row_idx * numel + col_idx] += uf_fac * (h_kron.data[i * 3 + j] - e[i] * e[j]);
                            } // for j
                        
                        } // for i
                    
                    } // for group_jdx
            
                } // for group_idx

            } // if k != 0

        } // for cell_idx

    // add up real-part and wave-part matrices
    ArrayHandle<Scalar> h_mobmat(m_mobmat,
                                 access_location::host,
                                 access_mode::overwrite); 
    for (unsigned int i = 0; i < numel * numel; i++)
        {
        h_mobmat.data[i] = h_mobmat_real.data[i] + h_mobmat_wave.data[i];
        }    
    
    } /* end of mobilityMatrix */

void TwoStepRPY::mobilityMatrixUF(const Scalar * fts,
                                  Scalar * uoe)
    {
    unsigned int group_size = m_group->getNumMembers();
    unsigned int numel = group_size * 3;

    ArrayHandle<Scalar> h_mobmat(m_mobmat,
                                 access_location::host,
                                 access_mode::read);

    cblas_dgemv(CblasRowMajor,
                CblasNoTrans,
                numel, numel,
                1.0,
                h_mobmat.data, numel,
                fts, 1,
                0.0,
                uoe, 1);

    } /* end of mobilityMatrixUF */

void TwoStepRPY::mobilityMatrixSqrtUF(const Scalar * psi,
                                      Scalar * uslip)
    {
        unsigned int group_size = m_group->getNumMembers();
        unsigned int numel = group_size * 3;

        ArrayHandle<Scalar> h_mobmat(m_mobmat,
                                    access_location::host,
                                    access_mode::read);
        ArrayHandle<Scalar> h_mobmat_scratch(m_mobmat_scratch,
                                             access_location::host,
                                             access_mode::overwrite);
        memcpy(h_mobmat_scratch.data, h_mobmat.data, numel * numel * sizeof(Scalar));

        ArrayHandle<Scalar> h_mobmatSqrt(m_mobmatSqrt,
                                         access_location::host,
                                         access_mode::overwrite);

        double * W = new double[numel];

        int INFO = LAPACKE_dsyev(LAPACK_ROW_MAJOR, 'V', 'U', numel, h_mobmat_scratch.data, numel, W);
        if (INFO != 0)
            {
            std::cout << "Error in dsyev: " << INFO << std::endl;
            exit(EXIT_FAILURE);
            }

        for (unsigned int i = 0; i < numel; i++)
            {
            W[i] = sqrt(W[i]);
            }

        for (unsigned int i = 0; i < numel; i++)
            {
            
            for (unsigned int j = 0; j < numel; j++)
                {
                
                Scalar sum = 0.0;
                for (unsigned int k = 0; k < numel; k++)
                    {
                    sum += h_mobmat_scratch.data[i * numel + k] * W[k] * h_mobmat_scratch.data[j * numel + k];
                    } // for k
                h_mobmatSqrt.data[i * numel + j] = sum;

                } // for j

            } // for i
        
        cblas_dgemv(CblasRowMajor,
                    CblasNoTrans,
                    numel, numel,
                    1.0,
                    h_mobmatSqrt.data, numel,
                    psi, 1,
                    0.0,
                    uslip, 1);

        delete [] W;
    } /* end of mobilityMatrixSqrtUF */

void TwoStepRPY::brownianParticleRNG(uint64_t timestep,
                                     Scalar * psi)
    {
    Scalar currentTemp = m_T->operator()(timestep);
    Scalar fac = fast::sqrt(6.0 * currentTemp / m_deltaT);

    unsigned int group_size = m_group->getNumMembers();
    uint16_t m_seed = m_sysdef->getSeed();

    ArrayHandle<unsigned int> h_tag(m_pdata->getTags(),
                                    access_location::host,
                                    access_mode::read);
    for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
        {
        unsigned int idx = m_group->getMemberIndex(group_idx);
        unsigned int ptag = h_tag.data[idx];

        unsigned int i3 = idx * 3;

        RandomGenerator rng(hoomd::Seed(RNGIdentifier::TwoStepBD, timestep, m_seed), hoomd::Counter(ptag));
        UniformDistribution<Scalar> uniform(Scalar(-1), Scalar(1));

        // force
        psi[i3    ] = fac * uniform(rng);
        psi[i3 + 1] = fac * uniform(rng);
        psi[i3 + 2] = fac * uniform(rng);
        } // for group_idx
    } /* end of brownianParticleRNG */

void TwoStepRPY::solverMobilityMatrixUF(uint64_t timestep,
                                        const Scalar * fts,
                                        Scalar * uoe)
    {
    mobilityMatrix();
    // deterministic velocity
    ArrayHandle<Scalar> h_u_real(m_u_real,
                                 access_location::host,
                                 access_mode::overwrite);
    mobilityMatrixUF(fts,
                     h_u_real.data);
    
    // random velocity
    ArrayHandle<Scalar> h_iter_psi(m_iter_psi,
                                   access_location::host,
                                   access_mode::overwrite);
    ArrayHandle<Scalar> h_uslip(m_uslip,
                                access_location::host,
                                access_mode::overwrite);
    brownianParticleRNG(timestep,
                        h_iter_psi.data);
    mobilityMatrixSqrtUF(h_iter_psi.data,
                         h_uslip.data);

    unsigned int group_size = m_group->getNumMembers();
    unsigned int numel = group_size * 3;

    for (unsigned int i = 0; i < numel; i++)
        {
        uoe[i] = h_uslip.data[i] + h_u_real.data[i];
        }
    
    
    } /* end of solverMobilityMatrixUF */

void TwoStepRPY::integrateStepOne(uint64_t timestep)
    {
    if (m_need_initialize)
        {
        setupMesh();
        initializeWorkArray();
        computeWaveValue();
        
        m_need_initialize = false;

        std::cout << "global mesh dimension: " << m_global_dim.x << " " << m_global_dim.y << " " << m_global_dim.z << std::endl;
        std::cout << "local mesh dimension: " << m_mesh_points.x << " " << m_mesh_points.y << " " << m_mesh_points.z << std::endl;
        std::cout << "ghost cells: " << m_n_ghost_cells.x << " " << m_n_ghost_cells.y << " " << m_n_ghost_cells.z << std::endl;
        std::cout << "grid dimension: " << m_grid_dim.x << " " << m_grid_dim.y << " " << m_grid_dim.z << std::endl;
        std::cout << "total number of grid " << m_NNN << std::endl;
        std::cout << "number of inner cells " << m_n_inner_cells << std::endl;
        std::cout << "number of ghost offset " << m_ghost_offset << std::endl;
        std::cout << "P = " << m_P << std::endl;
        }
    
    if (timestep % 100 == 0)
        {
        std::cout << "\nStep " << timestep << std::endl;
        }
    
    m_nlist->compute(timestep);
    // std::cout << "\nStep " << timestep << std::endl;

    bool ghost_cell_num_changed = false;
    uint3 n_ghost_cells = computeGhostCellNum();
    // do we need to reallocate?
    if (m_n_ghost_cells.x != n_ghost_cells.x || m_n_ghost_cells.y != n_ghost_cells.y || m_n_ghost_cells.z != n_ghost_cells.z)
        {
        ghost_cell_num_changed = true;
        }
    if (m_box_changed || ghost_cell_num_changed)
        {
        if (ghost_cell_num_changed)
            {
            setupMesh();
            }
        computeWaveValue();
        m_box_changed = false;
        }

    unsigned int group_size = m_group->getNumMembers();

    ArrayHandle<unsigned int> h_index_array(m_group->getIndexArray(),
                                            access_location::host,
                                            access_mode::read);
    ArrayHandle<unsigned int> h_nneigh(m_nlist->getNNeighArray(),
                                       access_location::host,
                                       access_mode::read);
    ArrayHandle<unsigned int> h_nlist(m_nlist->getNListArray(),
                                      access_location::host,
                                      access_mode::read);
    ArrayHandle<size_t> h_headlist(m_nlist->getHeadList(),
                                   access_location::host,
                                   access_mode::read);
    
    //~ Step 1 Calculate and collect all necessary forces    
    ArrayHandle<Scalar4> h_force(m_pdata->getNetForce(),
                                 access_location::host,
                                 access_mode::read);
    ArrayHandle<Scalar> h_fts(m_fts,
                              access_location::host,
                              access_mode::readwrite);
    
    for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
        {
        unsigned int idx = m_group->getMemberIndex(group_idx);
        unsigned int i3 = idx * 3;

        h_fts.data[i3    ] = h_force.data[idx].x;
        h_fts.data[i3 + 1] = h_force.data[idx].y;
        h_fts.data[i3 + 2] = h_force.data[idx].z;
        } // for group_idx;

    //~ Step 2 Solve for velocity of particles
    ArrayHandle<Scalar> h_uoe(m_uoe,
                              access_location::host,
                              access_mode::readwrite);

    solverMobilityUF(timestep,
                     h_fts.data,
                     h_uoe.data);
    // solverMobilityMatrixUF(timestep,
    //                        h_fts.data,
    //                        h_uoe.data);
    
    //~ Step 3 Map UOE to particle's velocity data
    ArrayHandle<Scalar4> h_vel(m_pdata->getVelocities(),
                               access_location::host,
                               access_mode::readwrite);
    for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
        {
        unsigned int idx = m_group->getMemberIndex(group_idx);
        unsigned int i3 = idx * 3;

        h_vel.data[idx].x = h_uoe.data[i3    ];
        h_vel.data[idx].y = h_uoe.data[i3 + 1];
        h_vel.data[idx].z = h_uoe.data[i3 + 2];
        } // for group_idx;
    
    //~ Step 4 Update particles' positions
    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(),
                               access_location::host,
                               access_mode::readwrite);
    ArrayHandle<int3> h_image(m_pdata->getImages(), 
                              access_location::host, 
                              access_mode::readwrite);
    const BoxDim& box = m_pdata->getBox();

    for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
        {
        unsigned int idx = m_group->getMemberIndex(group_idx);
        int3 img = h_image.data[idx];
        Scalar3 pos = make_scalar3(h_pos.data[idx].x, h_pos.data[idx].y, h_pos.data[idx].z);
        Scalar3 vel = make_scalar3(h_vel.data[idx].x, h_vel.data[idx].y, h_vel.data[idx].z);

        pos += vel * m_deltaT;
        box.wrap(pos, img);

        h_pos.data[idx].x = pos.x;
        h_pos.data[idx].y = pos.y;
        h_pos.data[idx].z = pos.z;
        }

    } /* end of integrateStepOne */

void TwoStepRPY::integrateStepTwo(uint64_t timstep)
    {
    // there is no step 2
    }

namespace detail
    {
void export_TwoStepRPY(pybind11::module& m)
    {
    pybind11::class_<TwoStepRPY, IntegrationMethodTwoStep, std::shared_ptr<TwoStepRPY>>(m, "TwoStepRPY")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>,
                            std::shared_ptr<ParticleGroup>,
                            std::shared_ptr<Variant>,
                            std::shared_ptr<NeighborList>,
                            Scalar,
                            Scalar>())
        .def("setParams", &TwoStepRPY::setParams);
    }

    } // end namespace detail

    } // end namespace md
    } // end namespace hoomd
