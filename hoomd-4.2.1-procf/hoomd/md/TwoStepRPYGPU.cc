#include "TwoStepRPYGPU.h"
#include "hoomd/HOOMDMath.h"
#include "hoomd/VectorMath.h"
#include "hoomd/GlobalArray.h"
#include <fstream>

#ifdef ENABLE_HIP
#include "TwoStepRPYGPU.cuh"

namespace hoomd
    {
namespace md
    {

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

TwoStepRPYGPU::TwoStepRPYGPU(std::shared_ptr<SystemDefinition> sysdef,
                             std::shared_ptr<ParticleGroup> group,
                             std::shared_ptr<Variant> T,
                             std::shared_ptr<NeighborList> nlist,
                             Scalar xi,
                             Scalar error) : TwoStepRPY(sysdef, group, T, nlist, xi, error), m_local_fft(true), m_block_size(256)
    {
    m_tuner_assign.reset(new Autotuner<1>({AutotunerBase::makeBlockSizeRange(m_exec_conf)}, m_exec_conf, "rpy_assign"));
    m_tuner_green.reset(new Autotuner<1>({AutotunerBase::makeBlockSizeRange(m_exec_conf)}, m_exec_conf, "rpy_green"));
    m_tuner_interpolate.reset(new Autotuner<1>({AutotunerBase::makeBlockSizeRange(m_exec_conf)}, m_exec_conf, "rpy_interpolate"));
    m_tuner_wavefunc.reset(new Autotuner<1>({AutotunerBase::makeBlockSizeRange(m_exec_conf)}, m_exec_conf, "rpy_wavefunc"));
    m_tuner_gridrng.reset(new Autotuner<1>({AutotunerBase::makeBlockSizeRange(m_exec_conf)}, m_exec_conf, "rpy_gridrng"));
    m_tuner_integrate.reset(new Autotuner<1>({AutotunerBase::makeBlockSizeRange(m_exec_conf)}, m_exec_conf, "rpy_integrate"));

    m_autotuners.insert(m_autotuners.end(), {m_tuner_assign, m_tuner_green, m_tuner_interpolate, m_tuner_wavefunc, m_tuner_gridrng, m_tuner_integrate});

    m_cufft_initialized = false;
    m_cuda_dfft_initialized = false;
    } /* end of constructor */

void TwoStepRPYGPU::setParams()
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

    std::vector<unsigned int> Mlist;
    for ( int ii = 0; ii < 10; ++ii ){
		int pow2 = 1;
		for ( int i = 0; i < ii; ++i ){
			pow2 *= 2;
		}
		for ( int jj = 0; jj < 6; ++jj ){
			int pow3 = 1;
			for ( int j = 0; j < jj; ++j ){
				pow3 *= 3;
			}
			for ( int kk = 0; kk < 4; ++kk ){
				int pow5 = 1;
				for ( int k = 0; k < kk; ++k ){
					pow5 *= 5;
				}
				int Mcurr = pow2 * pow3 * pow5;
				if ( Mcurr >= 8 && Mcurr <= 512 ){
					Mlist.push_back(Mcurr);
				}
			}
		}
	}
	std::sort(Mlist.begin(),Mlist.end());
	const unsigned int nmult = Mlist.size();

    for ( int ii = 0; ii < nmult; ++ii ){
		if (m_mesh_points.x <= Mlist[ii]){
			 m_mesh_points.x = Mlist[ii];
			break;
		}
	}
	for ( int ii = 0; ii < nmult; ++ii ){
		if (m_mesh_points.y <= Mlist[ii]){
			m_mesh_points.y = Mlist[ii];
			break;
		}
	}
	for ( int ii = 0; ii < nmult; ++ii ){
		if (m_mesh_points.z <= Mlist[ii]){
			m_mesh_points.z = Mlist[ii];
			break;
		}
	}
    

    m_global_dim = m_mesh_points;
    m_NNN = m_global_dim.x * m_global_dim.y * m_global_dim.z;

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
    std::cout << "total inner mesh = " << m_n_cells << std::endl;
    std::cout << "inner mesh = " << m_n_inner_cells << std::endl;
    std::cout << "P = " << m_P << std::endl;
    std::cout << "eta = " << m_eta << std::endl;

    } /* end of setParams */

TwoStepRPYGPU::~TwoStepRPYGPU()
    {
    if (m_local_fft && m_cufft_initialized)
        {
#ifdef __HIP_PLATFORM_HCC__
        hipfftDestroy(m_hipfft_plan); //CHECK_HIPFFT_ERROR(hipfftDestroy(m_hipfft_plan));
#else
        cufftDestroy(m_hipfft_plan); //CHECK_HIPFFT_ERROR(cufftDestroy(m_hipfft_plan));
#endif
        }
    } /* end of destructor */

void TwoStepRPYGPU::setupMesh()
    {
    m_n_ghost_cells = computeGhostCellNum();
    std::cout << "Set up mesh" << std::endl;
    // extra ghost cells are as wide as the inner cells

    const BoxDim& box = m_pdata->getBox();
    Scalar3 cell_width = box.getNearestPlaneDistance() / make_scalar3(m_mesh_points.x, m_mesh_points.y, m_mesh_points.z);
    m_ghost_width = cell_width * make_scalar3(m_n_ghost_cells.x, m_n_ghost_cells.y, m_n_ghost_cells.z);

    m_exec_conf->msg->notice(6) << "RPY: (Re-)allocating ghost layer (" << m_n_ghost_cells.x << ", " << m_n_ghost_cells.y << ", " << m_n_ghost_cells.z << ")" << std::endl;

    m_grid_dim = make_uint3(m_mesh_points.x + 2 * m_n_ghost_cells.x,
                            m_mesh_points.y + 2 * m_n_ghost_cells.y,
                            m_mesh_points.z + 2 * m_n_ghost_cells.z);
    m_n_cells = m_grid_dim.x * m_grid_dim.y * m_grid_dim.z;
    m_n_inner_cells = m_mesh_points.x * m_mesh_points.y * m_mesh_points.z;

    // allocate memory for wave functions
    GlobalArray<Scalar4> gridk(m_n_inner_cells, m_exec_conf);
    m_gridk.swap(gridk);

    GlobalArray<Scalar3> ymob(m_n_inner_cells, m_exec_conf);
    m_ymob.swap(ymob);

    GlobalArray<Scalar2> sf(m_n_inner_cells, m_exec_conf);
    m_sf.swap(sf);

    initializeFFT();
    } /* end of setupMesh */

uint3 TwoStepRPYGPU::computeGhostCellNum()
    {
    uint3 n_ghost_cells = make_uint3(0, 0, 0);
#ifdef ENABLE_MPI
    if (m_pdata->getDomainDecomposition())
        {
        Index3D did = m_pdata->getDomainDecomposition()->getDomainIndexer();
        n_ghost_cells.x = (did.getW() > 1) ? m_radius : 0;
        n_ghost_cells.y = (did.getD() > 1) ? m_radius : 0;
        n_ghost_cells.z = (did.getH() > 1) ? m_radius : 0;
        }
#endif

#ifdef ENABLE_MPI
    if (m_sysdef->isDomainDecomposed())
        {
        Scalar r_buff = m_nlist->getRBuff() / 2.0;
        const BoxDim& box = m_pdata->getBox();
        Scalar3 cell_width = box.getNearestPlaneDistance() / make_scalar3(m_mesh_points.x, m_mesh_points.y, m_mesh_points.z);

        if (n_ghost_cells.x)
            n_ghost_cells.x += (unsigned int) (r_buff / cell_width.x) + 1;
        if (n_ghost_cells.y)
            n_ghost_cells.y += (unsigned int) (r_buff / cell_width.y) + 1;
        if (n_ghost_cells.z)
            n_ghost_cells.z += (unsigned int) (r_buff / cell_width.z) + 1;
        }
#endif
    return n_ghost_cells;
    } /* end of computeGhostCellNum */

void TwoStepRPYGPU::initializeFFT()
    {
    std::cout << "\nInitialize FFT" << std::endl;
    // free plans if they have already been initialized
    if (m_local_fft && m_cufft_initialized)
        {
#ifdef __HIP_PLATFORM_HCC__
        hipfftDestroy(m_hipfft_plan); //CHECK_HIPFFT_ERROR(hipfftDestroy(m_hipfft_plan));
#else
        cufftDestroy(m_hipfft_plan); //CHECK_HIPFFT_ERROR(cufftDestroy(m_hipfft_plan));
#endif
        }


    if (m_local_fft)
        {
// create plan on every device
#ifdef __HIP_PLATFORM_HCC__
        std::cout << "HIP FFT plan created" << std::endl;
        hipfftPlan3d(&m_hipfft_plan,
                     m_mesh_points.z,
                     m_mesh_points.y,
                     m_mesh_points.x,
                     HIPFFT_C2C);
#else
        std::cout << "CUDA FFT plan created" << std::endl;
        cufftPlan3d(&m_hipfft_plan,
                    m_mesh_points.z,
                    m_mesh_points.y,
                    m_mesh_points.x,
                    CUFFT_C2C);
#endif
        m_cufft_initialized = true;
        }
    
    // allocate mesh and transformed mesh
    printf("FFT m_n_cells = %u\n", m_n_cells);
    GlobalArray<hipfftComplex> mesh_Fx(m_n_cells, m_exec_conf);
    m_mesh_Fx.swap(mesh_Fx);

    GlobalArray<hipfftComplex> mesh_Fy(m_n_cells, m_exec_conf);
    m_mesh_Fy.swap(mesh_Fy);

    GlobalArray<hipfftComplex> mesh_Fz(m_n_cells, m_exec_conf);
    m_mesh_Fz.swap(mesh_Fz);

    unsigned int inv_mesh_elements = m_n_cells;
    GlobalArray<hipfftComplex> mesh_inv_Fx(inv_mesh_elements, m_exec_conf);
    m_mesh_inv_Fx.swap(mesh_inv_Fx);

    GlobalArray<hipfftComplex> mesh_inv_Fy(inv_mesh_elements, m_exec_conf);
    m_mesh_inv_Fy.swap(mesh_inv_Fy);

    GlobalArray<hipfftComplex> mesh_inv_Fz(inv_mesh_elements, m_exec_conf);
    m_mesh_inv_Fz.swap(mesh_inv_Fz);
    
    } /* end of initializeFFT */

void TwoStepRPYGPU::initializeWorkArray()
    {
    std::cout << "Initialize Work Array" << std::endl;
    m_m_lanczos_ff = 2;
    m_mmax = 100;
    unsigned int group_size = m_group->getNumMembers();
    unsigned int numel = group_size * 3;

    GlobalArray<Scalar> iter_psi(numel, m_exec_conf);
    m_iter_psi.swap(iter_psi);

    GlobalArray<Scalar> iter_ff_v(numel, m_exec_conf);
    m_iter_ff_v.swap(iter_ff_v);

    GlobalArray<Scalar> iter_ff_vj(numel, m_exec_conf);
    m_iter_ff_vj.swap(iter_ff_vj);

    GlobalArray<Scalar> iter_ff_vjm1(numel, m_exec_conf);
    m_iter_ff_vjm1.swap(iter_ff_vjm1);

    GlobalArray<Scalar> iter_ff_uold(numel, m_exec_conf);
    m_iter_ff_uold.swap(iter_ff_uold);

    GlobalArray<Scalar> iter_ff_u(numel, m_exec_conf);
    m_iter_ff_u.swap(iter_ff_u);

    GlobalArray<Scalar> iter_ff_V(numel * (m_mmax + 1), m_exec_conf);
    m_iter_ff_V.swap(iter_ff_V);

    GlobalArray<Scalar> iter_Mpsi(numel, m_exec_conf);
    m_iter_Mpsi.swap(iter_Mpsi);

    GlobalArray<Scalar> Tm(m_mmax, m_exec_conf);
    m_Tm.swap(Tm);

    GlobalArray<Scalar> uslip(numel, m_exec_conf);
    m_uslip.swap(uslip);

    GlobalArray<Scalar> uslip_wave(numel, m_exec_conf);
    m_uslip_wave.swap(uslip_wave);

    GlobalArray<Scalar> u_real(numel, m_exec_conf);
    m_u_real.swap(u_real);

    GlobalArray<Scalar> u_wave(numel, m_exec_conf);
    m_u_wave.swap(u_wave);

    GlobalArray<Scalar> u_determin(numel, m_exec_conf);
    m_u_determin.swap(u_determin);

    GlobalArray<Scalar> uoe(numel, m_exec_conf);
    m_uoe.swap(uoe);

    GlobalArray<Scalar> fts(numel, m_exec_conf);
    m_fts.swap(fts);

    } /* end of initializeWorkArray */

void TwoStepRPYGPU::setupCublas()
    {
    std::cout << "Setup CUBLAS" << std::endl;
    cublasCreate(&m_blasHandle);
    } /* end of setupCublas */

void TwoStepRPYGPU::computeWaveValue()
    {
    ArrayHandle<Scalar4> d_gridk(m_gridk,
                                 access_location::device,
                                 access_mode::overwrite);
    ArrayHandle<Scalar3> d_ymob(m_ymob,
                                access_location::device,
                                access_mode::overwrite);
    ArrayHandle<Scalar2> d_sf(m_sf,
                              access_location::device,
                              access_mode::overwrite);

    unsigned int block_size = m_tuner_wavefunc->getParam()[0];
   
    m_tuner_wavefunc->begin();
    kernel::gpu_compute_wave_value(m_mesh_points,
                                   d_gridk.data,
                                   d_ymob.data,
                                   d_sf.data,
                                   m_pdata->getBox(),
                                   m_local_fft,
                                   m_xi2,
                                   m_eta,
                                   block_size);

    if (m_exec_conf->isCUDAErrorCheckingEnabled())
        {
        CHECK_CUDA_ERROR();
        }

    m_tuner_wavefunc->end();
    } /* end of computeWaveValue */

void TwoStepRPYGPU::mobilityRealUF(Scalar * force,
                                   Scalar * ureal)
    {
    ArrayHandle<Scalar4> d_postype(m_pdata->getPositions(),
                                   access_location::device,
                                   access_mode::read);
    ArrayHandle<unsigned int> d_index_array(m_group->getIndexArray(),
                                            access_location::device,
                                            access_mode::read);

    ArrayHandle<unsigned int> d_nneigh(m_nlist->getNNeighArray(),
                                       access_location::device,
                                       access_mode::read);
    ArrayHandle<unsigned int> d_nlist(m_nlist->getNListArray(),
                                      access_location::device,
                                      access_mode::read);
    ArrayHandle<size_t> d_headlist(m_nlist->getHeadList(),
                                   access_location::device,
                                   access_mode::read);
    
    unsigned int group_size = m_group->getNumMembers();

    kernel::gpu_mobility_real_uf(group_size,
                                d_postype.data,
                                d_index_array.data,
                                d_nneigh.data,
                                d_nlist.data,
                                d_headlist.data,
                                m_self_func,
                                m_xi,
                                m_pisqrt,
                                m_xi_pisqrt_inv,
                                m_rcut_ewald,
                                m_pdata->getBox(),
                                force,
                                ureal,
                                m_block_size);
    if (m_exec_conf->isCUDAErrorCheckingEnabled())
        {
        CHECK_CUDA_ERROR();
        }

    } /* end of mobilityRealUF */

void TwoStepRPYGPU::assignParticleForce(Scalar * force)
    {
    ArrayHandle<Scalar4> d_postype(m_pdata->getPositions(),
                                   access_location::device,
                                   access_mode::read);
    ArrayHandle<hipfftComplex> d_mesh_Fx(m_mesh_Fx,
                                         access_location::device,
                                         access_mode::overwrite);
    ArrayHandle<hipfftComplex> d_mesh_Fy(m_mesh_Fy,
                                         access_location::device,
                                         access_mode::overwrite);
    ArrayHandle<hipfftComplex> d_mesh_Fz(m_mesh_Fz,
                                         access_location::device,
                                         access_mode::overwrite);
    ArrayHandle<unsigned int> d_index_array(m_group->getIndexArray(),
                                            access_location::device,
                                            access_mode::read);
    unsigned int group_size = m_group->getNumMembers();

    m_tuner_assign->begin();
    unsigned int block_size = m_tuner_assign->getParam()[0];
    
    kernel::gpu_assign_particle_force(m_mesh_points,
                                      m_n_ghost_cells,
                                      m_grid_dim,
                                      group_size,
                                      d_index_array.data,
                                      d_postype.data,
                                      force,
                                      m_P,
                                      m_gauss_fac,
                                      m_gauss_exp,
                                      m_pdata->getBox(),
                                      m_h,
                                      d_mesh_Fx.data,
                                      d_mesh_Fy.data,
                                      d_mesh_Fz.data,
                                      block_size);

    if (m_exec_conf->isCUDAErrorCheckingEnabled())
        {
        CHECK_CUDA_ERROR();
        }
    m_tuner_assign->end();

    } /* end of assignParticleForce */

void TwoStepRPYGPU::forwardFFT()
    {
    ArrayHandle<hipfftComplex> d_mesh_Fx(m_mesh_Fx,
                                            access_location::device,
                                            access_mode::read);
    ArrayHandle<hipfftComplex> d_mesh_Fy(m_mesh_Fy,
                                            access_location::device,
                                            access_mode::read);
    ArrayHandle<hipfftComplex> d_mesh_Fz(m_mesh_Fz,
                                            access_location::device,
                                            access_mode::read);
#ifdef __HIP_PLATFORM_HCC__
    hipfftExecC2C(m_hipfft_plan,
                  d_mesh_Fx.data,
                  d_mesh_Fx.data,
                  HIPFFT_FORWARD);
    hipfftExecC2C(m_hipfft_plan,
                  d_mesh_Fy.data,
                  d_mesh_Fy.data,
                  HIPFFT_FORWARD);
    hipfftExecC2C(m_hipfft_plan,
                  d_mesh_Fz.data,
                  d_mesh_Fz.data,
                  HIPFFT_FORWARD);
#else
    cufftExecC2C(m_hipfft_plan,
                 d_mesh_Fx.data,
                 d_mesh_Fx.data,
                 CUFFT_FORWARD);
    cufftExecC2C(m_hipfft_plan,
                 d_mesh_Fy.data,
                 d_mesh_Fy.data,
                 CUFFT_FORWARD);
    cufftExecC2C(m_hipfft_plan,
                 d_mesh_Fz.data,
                 d_mesh_Fz.data,
                 CUFFT_FORWARD);
#endif
    } /* end of forwardFFT */

void TwoStepRPYGPU::meshGreen()
    {
    ArrayHandle<hipfftComplex> d_mesh_Fx(m_mesh_Fx,
                                         access_location::device,
                                         access_mode::read);
    ArrayHandle<hipfftComplex> d_mesh_Fy(m_mesh_Fy,
                                         access_location::device,
                                         access_mode::read);
    ArrayHandle<hipfftComplex> d_mesh_Fz(m_mesh_Fz,
                                         access_location::device,
                                         access_mode::read);
    ArrayHandle<hipfftComplex> d_mesh_inv_Fx(m_mesh_inv_Fx,
                                             access_location::device,
                                             access_mode::overwrite);
    ArrayHandle<hipfftComplex> d_mesh_inv_Fy(m_mesh_inv_Fy,
                                             access_location::device,
                                             access_mode::overwrite);
    ArrayHandle<hipfftComplex> d_mesh_inv_Fz(m_mesh_inv_Fz,
                                             access_location::device,
                                             access_mode::overwrite);

    ArrayHandle<Scalar4> d_gridk(m_gridk,
                                 access_location::device,
                                 access_mode::read);
    ArrayHandle<Scalar3> d_ymob(m_ymob,
                                access_location::device,
                                access_mode::read);
    ArrayHandle<Scalar2> d_sf(m_sf,
                             access_location::device,
                             access_mode::read);

    unsigned int block_size = m_tuner_green->getParam()[0];
    
    m_tuner_green->begin();
    kernel::gpu_mesh_green(m_n_inner_cells,
                           d_mesh_Fx.data,
                           d_mesh_Fy.data,
                           d_mesh_Fz.data,
                           d_gridk.data,
                           d_ymob.data,
                           d_sf.data,
                           d_mesh_inv_Fx.data,
                           d_mesh_inv_Fy.data,
                           d_mesh_inv_Fz.data,
                           block_size);
    if (m_exec_conf->isCUDAErrorCheckingEnabled())
        {
        CHECK_CUDA_ERROR();
        }
    m_tuner_green->end();

    } /* end of meshGreen */

void TwoStepRPYGPU::backwardFFT()
    {    
    // do local inverse FFT 
    ArrayHandle<hipfftComplex> d_mesh_inv_Fx(m_mesh_inv_Fx,
                                                access_location::device,
                                                access_mode::overwrite);
    ArrayHandle<hipfftComplex> d_mesh_inv_Fy(m_mesh_inv_Fy,
                                                access_location::device,
                                                access_mode::overwrite);
    ArrayHandle<hipfftComplex> d_mesh_inv_Fz(m_mesh_inv_Fz,
                                                access_location::device,
                                                access_mode::overwrite);
    // do inverse FFT in-place
    m_exec_conf->beginMultiGPU();
#ifdef __HIP_PLATFORM_HCC__
    hipfftExecC2C(m_hipfft_plan,
                  d_mesh_inv_Fx.data,
                  d_mesh_inv_Fx.data,
                  HIPFFT_BACKWARD);
    hipfftExecC2C(m_hipfft_plan,
                  d_mesh_inv_Fy.data,
                  d_mesh_inv_Fy.data,
                  HIPFFT_BACKWARD);
    hipfftExecC2C(m_hipfft_plan,
                  d_mesh_inv_Fz.data,
                  d_mesh_inv_Fz.data,
                  HIPFFT_BACKWARD);
#else
    cufftExecC2C(m_hipfft_plan,
                 d_mesh_inv_Fx.data,
                 d_mesh_inv_Fx.data,
                 CUFFT_INVERSE);
    cufftExecC2C(m_hipfft_plan,
                 d_mesh_inv_Fy.data,
                 d_mesh_inv_Fy.data,
                 CUFFT_INVERSE);
    cufftExecC2C(m_hipfft_plan,
                 d_mesh_inv_Fz.data,
                 d_mesh_inv_Fz.data,
                 CUFFT_INVERSE);
#endif
    m_exec_conf->endMultiGPU();
        
    } /* end of backwardFFT */

void TwoStepRPYGPU::interpolateParticleVelocity(Scalar * uwave)
    {
    ArrayHandle<Scalar4> d_postype(m_pdata->getPositions(),
                                   access_location::device,
                                   access_mode::read);
    ArrayHandle<hipfftComplex> d_mesh_inv_Fx(m_mesh_inv_Fx,
                                             access_location::device,
                                             access_mode::read);
    ArrayHandle<hipfftComplex> d_mesh_inv_Fy(m_mesh_inv_Fy,
                                             access_location::device,
                                             access_mode::read);
    ArrayHandle<hipfftComplex> d_mesh_inv_Fz(m_mesh_inv_Fz,
                                             access_location::device,
                                             access_mode::read);
    ArrayHandle<unsigned int> d_index_array(m_group->getIndexArray(),
                                            access_location::device,
                                            access_mode::read);
    
    unsigned int block_size = m_tuner_interpolate->getParam()[0];

    m_tuner_interpolate->begin();
    kernel::gpu_interpolate_particle_velocity(m_mesh_points,
                                              m_n_ghost_cells,
                                              m_grid_dim,
                                              m_pdata->getN(),
                                              d_index_array.data,
                                              d_postype.data,
                                              m_P,
                                              m_gauss_fac,
                                              m_gauss_exp,
                                              m_vk,
                                              m_pdata->getBox(),
                                              m_h,
                                              m_local_fft,
                                              m_n_cells,
                                              d_mesh_inv_Fx.data,
                                              d_mesh_inv_Fy.data,
                                              d_mesh_inv_Fz.data,
                                              uwave,
                                              block_size);

    if (m_exec_conf->isCUDAErrorCheckingEnabled())
        {
        CHECK_CUDA_ERROR();
        }
    m_tuner_interpolate->end();
    } /* end of interpolateParticleVelocity */

void TwoStepRPYGPU::mobilityWaveUF(Scalar * force,
                                   Scalar * uwave)
    {
    // assign force
    assignParticleForce(force);

    // FFT
    forwardFFT();

    // scale by Green's function
    meshGreen();

    // IFFT
    backwardFFT();

    // interpolate velocity
    interpolateParticleVelocity(uwave);

    } /* end of mobilityWaveUF */

void TwoStepRPYGPU::mobilityGeneralUF(Scalar * force,
                                      Scalar * velocity)
    {
    unsigned int group_size = m_group->getNumMembers();
    ArrayHandle<unsigned int> d_index_array(m_group->getIndexArray(),
                                            access_location::device,
                                            access_mode::read);

    // obtain real-part velocity
    ArrayHandle<Scalar> d_u_real(m_u_real,
                                 access_location::device,
                                 access_mode::readwrite);
    mobilityRealUF(force,
                   d_u_real.data);
    
    // obtain wave-part velocity
    ArrayHandle<Scalar> d_u_wave(m_u_wave,
                                 access_location::device,
                                 access_mode::readwrite);
    mobilityWaveUF(force,
                   d_u_wave.data);

    // add two parts of velocities together
    kernel::gpu_mobility_velocity_sum(group_size,
                                      d_index_array.data,
                                      d_u_real.data,
                                      d_u_wave.data,
                                      velocity,
                                      m_block_size);

    if (m_exec_conf->isCUDAErrorCheckingEnabled())
        {
        CHECK_CUDA_ERROR();
        }

    } /* end of mobilityGeneralUF */

void TwoStepRPYGPU::brownianFarFieldSlipVelocityReal(uint64_t timestep,
                                                     Scalar * uslip_real)
    {
    // std::cout << "\nCC: Slip Velocity Real" << std::endl;
    // obtain random vector of particles
    uint16_t m_seed = m_sysdef->getSeed();
    Scalar currentTemp = m_T->operator()(timestep);
    unsigned int group_size = m_group->getNumMembers();

    ArrayHandle<unsigned int> d_index_array(m_group->getIndexArray(),
                                            access_location::device,
                                            access_mode::read);
    ArrayHandle<unsigned int> d_tag(m_pdata->getTags(),
                                    access_location::device,
                                    access_mode::read);
    ArrayHandle<Scalar> d_iter_psi(m_iter_psi,
                                   access_location::device,
                                   access_mode::readwrite);
    kernel::gpu_brownian_farfield_particle_rng(currentTemp,
                                               timestep,
                                               m_seed,
                                               m_deltaT,
                                               group_size,
                                               d_index_array.data,
                                               d_tag.data,
                                               d_iter_psi.data,
                                               m_block_size);
    if (m_exec_conf->isCUDAErrorCheckingEnabled())
        {
        CHECK_CUDA_ERROR();
        }

    // calculate U = M^0.5 . Psi by using Lanczos algorithm
    ArrayHandle<Scalar> d_iter_ff_v(m_iter_ff_v,
                                    access_location::device,
                                    access_mode::readwrite);
    ArrayHandle<Scalar> d_iter_ff_vj(m_iter_ff_vj,
                                     access_location::device,
                                     access_mode::readwrite);
    ArrayHandle<Scalar> d_iter_ff_vjm1(m_iter_ff_vjm1,
                                       access_location::device,
                                       access_mode::readwrite);
    ArrayHandle<Scalar> d_iter_ff_uold(m_iter_ff_uold,
                                       access_location::device,
                                       access_mode::readwrite);
    ArrayHandle<Scalar> d_iter_ff_V(m_iter_ff_V,
                                    access_location::device,
                                    access_mode::readwrite);
    ArrayHandle<Scalar> d_iter_Mpsi(m_iter_Mpsi,
                                    access_location::device,
                                    access_mode::readwrite);
    ArrayHandle<Scalar> d_Tm(m_Tm,
                             access_location::device,
                             access_mode::readwrite);
    ArrayHandle<Scalar4> d_postype(m_pdata->getPositions(),
                                   access_location::device,
                                   access_mode::read);
    ArrayHandle<unsigned int> d_nneigh(m_nlist->getNNeighArray(),
                                       access_location::device,
                                       access_mode::read);
    ArrayHandle<unsigned int> d_nlist(m_nlist->getNListArray(),
                                      access_location::device,
                                      access_mode::read);
    ArrayHandle<size_t> d_headlist(m_nlist->getHeadList(),
                                   access_location::device,
                                   access_mode::read);

    kernel::gpu_brownian_lanczos(group_size,
                                 d_postype.data,
                                 d_index_array.data,
                                 d_nneigh.data,
                                 d_nlist.data,
                                 d_headlist.data,
                                 m_self_func,
                                 m_xi,
                                 m_pisqrt,
                                 m_xi_pisqrt_inv,
                                 m_rcut_ewald,
                                 m_pdata->getBox(),
                                 m_m_lanczos_ff,
                                 m_mmax,
                                 m_error,
                                 d_iter_psi.data,
                                 d_iter_ff_v.data,
                                 d_iter_ff_vj.data,
                                 d_iter_ff_vjm1.data,
                                 d_iter_ff_uold.data,
                                 d_iter_ff_V.data,
                                 d_iter_Mpsi.data,
                                 d_Tm.data,
                                 uslip_real,
                                 m_blasHandle,
                                 m_block_size);

    } /* end of brownianFarFieldSlipVelocityReal */

void TwoStepRPYGPU::brownianFarFieldSlipVelocityWave(uint64_t timestep,
                                                     Scalar * uslip_wave)
    {
    // generate random number on grid
    ArrayHandle<hipfftComplex> d_mesh_inv_Fx(m_mesh_inv_Fx,
                                             access_location::device,
                                             access_mode::readwrite);
    ArrayHandle<hipfftComplex> d_mesh_inv_Fy(m_mesh_inv_Fy,
                                             access_location::device,
                                             access_mode::readwrite);
    ArrayHandle<hipfftComplex> d_mesh_inv_Fz(m_mesh_inv_Fz,
                                             access_location::device,
                                             access_mode::readwrite);
    ArrayHandle<Scalar4> d_gridk(m_gridk,
                                 access_location::device,
                                 access_mode::read);
    ArrayHandle<Scalar3> d_ymob(m_ymob,
                                access_location::device,
                                access_mode::read);
    ArrayHandle<Scalar2> d_sf(m_sf,
                              access_location::device,
                              access_mode::read);
    uint16_t m_seed = m_sysdef->getSeed();
    Scalar currentTemp = m_T->operator()(timestep);

    unsigned int block_size = m_tuner_gridrng->getParam()[0];
    
    kernel::gpu_brownian_farfield_grid_rng(m_n_inner_cells,
                                           timestep,
                                           m_seed,
                                           m_mesh_points,
                                           currentTemp,
                                           m_vk,
                                           m_deltaT,
                                           d_gridk.data,
                                           d_ymob.data,
                                           d_sf.data,
                                           d_mesh_inv_Fx.data,
                                           d_mesh_inv_Fy.data,
                                           d_mesh_inv_Fz.data,
                                           block_size);
    if (m_exec_conf->isCUDAErrorCheckingEnabled())
        {
        CHECK_CUDA_ERROR();
        }

    m_tuner_gridrng->end();

    // IFFT
    backwardFFT();

    // interpolate velocity
    interpolateParticleVelocity(uslip_wave);

    } /* end of brownianFarFieldSlipVelocityWave */

void TwoStepRPYGPU::brownianFarFieldSlipVelocity(uint64_t timestep,
                                                 Scalar * uslip)
    {
    // wave part of slip velocity
    ArrayHandle<Scalar> d_uslip_wave(m_uslip_wave,
                                     access_location::device,
                                     access_mode::readwrite);
    brownianFarFieldSlipVelocityWave(timestep,
                                     d_uslip_wave.data);
    
    // real part of slip velocity
    ArrayHandle<Scalar> d_iter_ff_u(m_iter_ff_u,
                                    access_location::device,
                                    access_mode::readwrite);
    brownianFarFieldSlipVelocityReal(timestep,
                                     d_iter_ff_u.data);

    // add up slip velocity
    unsigned int group_size = m_group->getNumMembers();
    ArrayHandle<unsigned int> d_index_array(m_group->getIndexArray(),
                                            access_location::device,
                                            access_mode::read);
    kernel::gpu_mobility_velocity_sum(group_size,
                                      d_index_array.data,
                                      d_uslip_wave.data,
                                      d_iter_ff_u.data,
                                      uslip,
                                      m_block_size);

    } /* end of brownianFarFieldSlipVelocity */

void TwoStepRPYGPU::solverMobilityUF(uint64_t timestep,
                                     Scalar * force,
                                     Scalar * uoe)
    {
    // calculate deterministic velocity
    ArrayHandle<Scalar> d_u_determin(m_u_determin,
                                     access_location::device,
                                     access_mode::readwrite);
    mobilityGeneralUF(force,
                      d_u_determin.data);

    // calculate stochastic velocity
    Scalar currentTemp = m_T->operator()(timestep);
    if (currentTemp > 0.0)
        {
        ArrayHandle<Scalar> d_uslip(m_uslip,
                                    access_location::device,
                                    access_mode::readwrite);
        brownianFarFieldSlipVelocity(timestep,
                                     d_uslip.data);
       
        // add up velocity
        unsigned int group_size = m_group->getNumMembers();
        ArrayHandle<unsigned int> d_index_array(m_group->getIndexArray(),
                                                access_location::device,
                                                access_mode::read);
        kernel::gpu_mobility_velocity_sum(group_size,
                                          d_index_array.data,
                                          d_u_determin.data,
                                          d_uslip.data,
                                          uoe,
                                          m_block_size);
        }

    } /* end of solverMobilityUF */

void TwoStepRPYGPU::integrateStepOne(uint64_t timestep)
    {
    unsigned int group_size = m_group->getNumMembers();

    if (m_need_initialize)
        {
        setupMesh();
        initializeWorkArray();
        setupCublas();
        computeWaveValue();

        m_need_initialize = false;
        }

    // printf("\nStep %lu\n", timestep);
    if (m_box_changed)
        {
        // printf("Box changed\n");
        computeWaveValue();
        m_box_changed = false;
        }
    else
        {
        // printf("Box not changed\n");
        }
    
    m_nlist->compute(timestep);
    if (timestep % 1000 == 0)
        {
        std::cout << "Step " << timestep << std::endl;
        }
    
    // collect force of particles and convert Scalar4 to Scalar
    ArrayHandle<Scalar4> d_netforce(m_pdata->getNetForce(),
                                    access_location::device,
                                    access_mode::read);
    ArrayHandle<Scalar> d_fts(m_fts,
                              access_location::device,
                              access_mode::readwrite);
    ArrayHandle<unsigned int> d_index_array(m_group->getIndexArray(),
                                            access_location::device,
                                            access_mode::read);
    kernel::gpu_convert_scalar4_to_scalar(group_size,
                                          d_index_array.data,
                                          d_netforce.data,
                                          d_fts.data,
                                          m_block_size);
    // calculate velocity
    ArrayHandle<Scalar> d_uoe(m_uoe,
                              access_location::device,
                              access_mode::readwrite);
    solverMobilityUF(timestep,
                     d_fts.data,
                     d_uoe.data);

    // convert velocity of Scalar to Scalar4
    ArrayHandle<Scalar4> d_totvelocity(m_pdata->getVelocities(),
                                       access_location::device,
                                       access_mode::readwrite);
    kernel::gpu_convert_scalar_to_scalar4(group_size,
                                          d_index_array.data,
                                          d_uoe.data,
                                          d_totvelocity.data,
                                          m_block_size);

    // update position
    Scalar shear_rate = this->m_SR;
    ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(),
                               access_location::device,
                               access_mode::readwrite);
    ArrayHandle<int3> d_image(m_pdata->getImages(),
                              access_location::device,
                              access_mode::readwrite);
    m_exec_conf->beginMultiGPU();
    m_tuner_integrate->begin();
    kernel::gpu_rpy_step_one(group_size,
                             d_index_array.data,
                             m_pdata->getBox(),
                             shear_rate,
                             d_image.data,
                             m_deltaT,
                             d_totvelocity.data,
                             d_pos.data,
                             m_block_size,
                             m_group->getGPUPartition());

    if (m_exec_conf->isCUDAErrorCheckingEnabled())
        {
        CHECK_CUDA_ERROR();
        }
    m_tuner_integrate->end();
    m_exec_conf->endMultiGPU();

    } /* end of integrateStepOne */

void TwoStepRPYGPU::integrateStepTwo(uint64_t timestep)
    {
    // there is no step 2
    }

namespace detail
    {
void export_TwoStepRPYGPU(pybind11::module& m)
    {
    pybind11::class_<TwoStepRPYGPU, TwoStepRPY, std::shared_ptr<TwoStepRPYGPU>>(m, "TwoStepRPYGPU")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>,
                            std::shared_ptr<ParticleGroup>,
                            std::shared_ptr<Variant>,
                            std::shared_ptr<NeighborList>,
                            Scalar,
                            Scalar>())
        .def("setParams", &TwoStepRPYGPU::setParams);
    }

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd


#endif
