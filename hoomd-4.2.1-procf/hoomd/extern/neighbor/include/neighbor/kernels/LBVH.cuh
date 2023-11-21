// Copyright (c) 2018-2019, Michael P. Howard.
// This file is released under the Modified BSD License.

// Maintainer: mphoward

#ifndef NEIGHBOR_KERNELS_LBVH_CUH_
#define NEIGHBOR_KERNELS_LBVH_CUH_

#include <hipper/hipper_runtime.h>
#include <hipper/hipper_cub.h>

#include "../BoundingVolumes.h"
#include "../LBVHData.h"

namespace neighbor
{
namespace gpu
{
namespace kernel
{
//! Expand a 10-bit integer into 30 bits by inserting 2 zeros after each bit.
/*!
 * \param v unsigned integer with 10 bits set
 * \returns The integer expanded with two zeros interleaved between bits
 * http://devblogs.nvidia.com/parallelforall/thinking-parallel-part-iii-tree-construction-gpu/
 */
__device__ __forceinline__ unsigned int expandBits(unsigned int v)
{
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

//! Compute the 30-bit Morton code for a tuple of binned indexes.
/*!
 * \param point (x,y,z) tuple of bin indexes.
 * \returns 30-bit Morton code corresponding to \a point.
 *
 * The Morton code is formed by first expanding the bits of each component (see ::expandBits),
 * and then bitshifting to interleave them. The Morton code then has a representation::
 *
 *  x0y0z0x1y1z1...
 *
 * where indices refer to the bitwise representation of each component.
 */
__device__ __forceinline__ unsigned int calcMortonCode(uint3 point)
    {
    return 4 * expandBits(point.x) + 2 * expandBits(point.y) + expandBits(point.z);
    }

//! Compute the number of bits shared by Morton codes for primitives \a i and \a j.
/*!
 * \param d_codes List of Morton codes.
 * \param code_i Morton code corresponding to \a i.
 * \param i First primitive.
 * \param j Second primitive.
 * \param N Number of primitives.
 *
 * \returns Number of bits in longest common prefix or -1 if \a j lies outside [0,N).
 *
 * The longest common prefix of the Morton codes for \a i and \j is computed
 * using the __clz intrinsic. When \a i and \a j are the same, they share all 32
 * bits in the int representation of the Morton code. In that case, the common
 * prefix of \a i and \a j is used as a tie breaker.
 *
 * The user is required to supply \a code_i (even though it could also be looked
 * up from \a d_codes) for performance reasons, since code_i can be cached by
 * the caller if making multiple calls to ::delta for different \a j.
 */
__device__ __forceinline__ int delta(const unsigned int *d_codes,
                                     const unsigned int code_i,
                                     const int i,
                                     const int j,
                                     const unsigned int N)
    {
    if (j < 0 || j >= N)
        {
        return -1;
        }

    const unsigned int code_j = d_codes[j];

    if (code_i == code_j)
        {
        return (32 + __clz(i ^ j));
        }
    else
        {
        return __clz(code_i ^ code_j);
        }
    }

//! Convert a fraction to [0,1023]
/*
 * \param f Fractional coordinate lying in [0,1].
 * \returns Bin integer lying in [0,1023]
 *
 * The range of the binned integer corresponds to the maximum value that can be
 * stored in a 10-bit integer. When \a f lies outside [0,1], the bin is clamped to
 * the ends of the range.
 */
__device__ __forceinline__ unsigned int fractionToBin(float f)
    {
    return static_cast<unsigned int>(fminf(fmaxf(f * 1023.f, 0.f), 1023.f));
    }

//! Kernel to generate the Morton codes
/*!
 * \param d_codes Generated Morton codes.
 * \param d_indexes Generated index for the primitive.
 * \param insert Insert operation.
 * \param lo Lower bound of scene.
 * \param hi Upper bound of scene.
 * \param N Number of primitives.
 *
 * \tparam InsertOpT the kind of insert operation
 *
 * One thread is used to process each primitive. The point is binned into
 * one of 2^10 bins using its fractional coordinate between \a lo and \a hi.
 * The bins are converted to a Morton code. The Morton code and corresponding
 * primitive index are stored. The reason for storing the primitive index now
 * is for subsequent sorting (see ::lbvh_sort_codes).
 */
template<class InsertOpT>
__global__ void lbvh_gen_codes(unsigned int *d_codes,
                               unsigned int *d_indexes,
                               const InsertOpT insert,
                               const float3 lo,
                               const float3 hi,
                               const unsigned int N)
    {
    // one thread per point
    const unsigned int idx = hipper::threadRank<1,1>();
    if (idx >= N)
        return;

    // real space coordinate of aabb center
    const float3 r = insert.get(idx).getCenter();

    // fractional coordinate
    const float3 f = make_float3((r.x - lo.x) / (hi.x - lo.x),
                                 (r.y - lo.y) / (hi.y - lo.y),
                                 (r.z - lo.z) / (hi.z - lo.z));

    // bin fractional coordinate
    const uint3 q = make_uint3(fractionToBin(f.x), fractionToBin(f.y), fractionToBin(f.z));

    // compute morton code
    const unsigned int code = calcMortonCode(q);

    // write out morton code and primitive index
    d_codes[idx] = code;
    d_indexes[idx] = idx;
    }

//! Kernel to generate the tree hierarchy
/*!
 * \param tree LBVH tree (raw pointers)
 * \param d_codes Sorted Morton codes for the primitives.
 * \param N Number of primitives
 *
 * One thread is used per *internal* node. (The LBVH guarantees that there are
 * exactly N-1 internal nodes.) The algorithm is given by Figure 4 of
 * <a href="https://dl.acm.org/citation.cfm?id=2383801">Karras</a>.
 */
__global__ static void lbvh_gen_tree(const LBVHData tree,
                                     const unsigned int *d_codes,
                                     const unsigned int N)
    {
    // one thread per internal node (= N-1 threads)
    const unsigned int i = hipper::threadRank<1,1>();
    if (i >= N-1)
        return;

    const unsigned int code_i = d_codes[i];
    const int forward_prefix = delta(d_codes, code_i, i, i+1, N);
    const int backward_prefix = delta(d_codes, code_i, i, i-1, N);

    // get direction of the range based on sign
    const int d = (forward_prefix >= backward_prefix) - (forward_prefix < backward_prefix);

    // get minimum prefix
    const int min_prefix = delta(d_codes, code_i, i, i-d, N);

    // get maximum prefix by binary search
    int lmax = 2;
    while( delta(d_codes, code_i, i, i + d*lmax, N) > min_prefix)
        {
        lmax = lmax << 1;
        }
    int l = 0; int t = lmax;
    do
        {
        t = t >> 1;
        if (delta(d_codes, code_i, i, i + (l+t)*d, N) > min_prefix)
            l = l + t;
        }
    while (t > 1);
    const int j = i + l*d;

    // get the length of the common prefix
    const int common_prefix = delta(d_codes, code_i, i, j, N);

    // binary search to find split position
    int s = 0; t = l;
    do
        {
        t = (t + 1) >> 1;
        // if proposed split lies within range
        if (s+t < l)
            {
            const int split_prefix = delta(d_codes, code_i, i, i+(s+t)*d, N);

            // if new split shares a longer number of bits, accept it
            if (split_prefix > common_prefix)
                {
                s = s + t;
                }
            }
        }
    while (t > 1);
    const int split = i + s*d + min(d,0);

    const int left = (min(i,j) == split) ? split + (N-1) : split;
    const int right = (max(i,j) == (split + 1)) ? split + N : split + 1;

    // children
    tree.left[i] = left;
    tree.right[i] = right;

    // parents
    tree.parent[left] = i;
    tree.parent[right] = i;

    // root node (index 0) has no parent
    if (i == 0)
        {
        tree.parent[0] = LBVHSentinel;
        }
    }

//! Bubble the bounding boxes up the tree hierarchy.
/*!
 * \param tree LBVH tree (raw pointers).
 * \param insert the insert operation to get the aabbs
 * \param d_locks Temporary storage for state of internal nodes.
 * \param N Number of primitives.
 *
 * \tparam InsertOpT the kind of insert operation
 *
 * One thread originally processes each primitive. The primitive is represented by a BoundingBox
 * defined using two float3s. This bounding box is stored for the leaf. Then, each thread
 * begins to process up the tree hierarchy.
 *
 * The second thread to reach each node processes the node, which ensures that all children
 * have already been processed. The order to reach the node is determined by an atomic
 * operation on \a d_locks. The bounding box of the node being processed is determined by
 * merging the bounding box of the child processing its parent with the bounding box of its
 * sibling. The process is then repeated until the root node is reached.
 *
 * The InsertOpT is used to determine initial bounding boxes for the primitives.
 *
 * \note
 * A __threadfence() is employed after the AABB is stored to ensure that it is visible to
 * other threads reading from global memory.
 */
template<class InsertOpT>
__global__ void lbvh_bubble_aabbs(const LBVHData tree,
                                  const InsertOpT insert,
                                  unsigned int *d_locks,
                                  const unsigned int N)
    {
    // one thread per point
    const unsigned int idx = hipper::threadRank<1,1>();
    if (idx >= N)
        return;

    // determine lower and upper bounds of the primitive, even in mixed precision
    BoundingBox b = insert.get(tree.primitive[idx]);
    float3 lo = b.lo;
    float3 hi = b.hi;

    // set aabb for the leaf node
    int last = N-1+idx;
    tree.lo[last] = lo;
    tree.hi[last] = hi;
    __threadfence();

    int current = tree.parent[last];
    while (current != LBVHSentinel)
        {
        // parent is processed by the second thread to reach it
        unsigned int lock = atomicAdd(d_locks + current, 1);
        if (!lock)
            return;

        // look for the sibling of the current thread with speculation
        int sibling = tree.left[current];
        if (sibling == last)
            {
            sibling = tree.right[current];
            }

        // compute min / max bounds of the current thread with its sibling
        const float3 sib_lo = tree.lo[sibling];
        if (sib_lo.x < lo.x) lo.x = sib_lo.x;
        if (sib_lo.y < lo.y) lo.y = sib_lo.y;
        if (sib_lo.z < lo.z) lo.z = sib_lo.z;

        const float3 sib_hi = tree.hi[sibling];
        if (sib_hi.x > hi.x) hi.x = sib_hi.x;
        if (sib_hi.y > hi.y) hi.y = sib_hi.y;
        if (sib_hi.z > hi.z) hi.z = sib_hi.z;

        // write out bounding box to global memory
        tree.lo[current] = lo;
        tree.hi[current] = hi;
        __threadfence();

        // move up tree
        last = current;
        current = tree.parent[current];
        }
    }

//! Kernel to set data for a one-primitive LBVH.
/*!
 * \param tree LBVH tree (raw pointers).
 * \param insert The insert operation to obtain the ONE aabb.
 *
 * \tparam InsertOpT the kind of insert operation.
 *
 * The one-primitive LBVH needs to be handled in a special kernel
 * because it violates assumptions about how to traverse the tree,
 * and this would be overkill anyway.
 */
template<class InsertOpT>
__global__ void lbvh_one_primitive(const LBVHData tree,
                                   const InsertOpT insert)
    {
    // one thread only
    const unsigned int idx = hipper::threadRank<1,1>();
    if (idx >= 1)
        return;

    const BoundingBox b = insert.get(0);

    tree.parent[0] = LBVHSentinel;
    tree.lo[0] = b.lo;
    tree.hi[0] = b.hi;
    }

} // end namespace kernel

//! Generate Morton codes for the primitives.
/*!
 * \param d_codes Generated Morton codes.
 * \param d_indexes Generated index for the primitive.
 * \param insert Insert operation.
 * \param lo Lower bound of scene.
 * \param hi Upper bound of scene.
 * \param N Number of primitives.
 * \param block_size Number of CUDA threads per block.
 * \param stream CUDA stream for kernel execution.
 *
 * \tparam InsertOpT the kind of insert operation
 *
 * \sa kernel::lbvh_gen_codes
 */
template<class InsertOpT>
void lbvh_gen_codes(unsigned int *d_codes,
                    unsigned int *d_indexes,
                    const InsertOpT& insert,
                    const float3 lo,
                    const float3 hi,
                    const unsigned int N,
                    const unsigned int block_size,
                    hipper::stream_t stream)
    {
    // clamp block size
    static unsigned int max_block_size = UINT_MAX;
    if (max_block_size == UINT_MAX)
        {
        hipper::funcAttributes_t attr;
        hipper::funcGetAttributes(&attr, reinterpret_cast<const void*>(kernel::lbvh_gen_codes<InsertOpT>));
        max_block_size = attr.maxThreadsPerBlock;
        }
    const unsigned int run_block_size = (block_size < max_block_size) ? block_size : max_block_size;
    const unsigned int num_blocks = (N + run_block_size - 1)/run_block_size;

    hipper::KernelLauncher launcher(num_blocks, run_block_size, stream);
    launcher(kernel::lbvh_gen_codes<InsertOpT>, d_codes, d_indexes, insert, lo, hi, N);
    }

//! Sort the primitives into Morton code order.
/*!
 * \param d_tmp Temporary storage for CUB.
 * \param tmp_bytes Temporary storage size (B) for CUB.
 * \param d_codes Unsorted Morton codes.
 * \param d_alt_codes Alternate array of Morton codes.
 * \param d_indexes Unsorted primitive indexes.
 * \param d_alt_indexes Alternate array of primitive indexes.
 * \param N Number of primitives.
 * \param stream CUDA stream for kernel execution.
 *
 * \returns Two flags (swap) with the location of the sorted codes and indexes. If swap.x
 *          is 1, then the sorted codes are in \a d_alt_codes and need to be swapped. Similarly,
 *          if swap.y is 1, then the sorted indexes are in \a d_alt_indexes.
 *
 * The Morton codes are sorted in ascending order using radix sort in the CUB library.
 * This function must be called twice in order for the sort to occur. When \a d_tmp is NULL
 * on the first call, CUB sizes the temporary storage that is required and sets it in \a tmp_bytes.
 * Some versions of CUB were buggy and required \a d_tmp be allocated even
 * when \a tmp_bytes was 0. To bypass this, allocate a small amount (say, 4B) when \a tmp_bytes is 0.
 * The second call will then sort the Morton codes and indexes. The sorted data will be in the
 * appropriate buffer, which can be determined by the returned flags.
 */
inline uchar2 lbvh_sort_codes(void *d_tmp,
                              size_t &tmp_bytes,
                              unsigned int *d_codes,
                              unsigned int *d_alt_codes,
                              unsigned int *d_indexes,
                              unsigned int *d_alt_indexes,
                              const unsigned int N,
                              hipper::stream_t stream)
    {
    hipper::cub::DoubleBuffer<unsigned int> d_keys(d_codes, d_alt_codes);
    hipper::cub::DoubleBuffer<unsigned int> d_vals(d_indexes, d_alt_indexes);

    hipper::cub::DeviceRadixSort::SortPairs(d_tmp, tmp_bytes, d_keys, d_vals, N, 0, 30, stream);

    // mark that the arrays should be flipped if the final result is not in the primary array
    uchar2 swap = make_uchar2(0,0);
    if (d_tmp != NULL)
        {
        swap.x = (d_keys.selector == 1);
        swap.y = (d_vals.selector == 1);
        }
    return swap;
    }

//! Generate the tree hierarchy from the Morton codes.
/*!
 * \param tree LBVH tree (raw pointers).
 * \param d_codes Sorted Morton codes for the primitives.
 * \param N Number of primitives.
 * \param block_size Number of CUDA threads per block.
 * \param stream CUDA stream for kernel execution.
 *
 * \sa kernel::lbvh_gen_tree
 */
inline void lbvh_gen_tree(const LBVHData tree,
                          const unsigned int *d_codes,
                          const unsigned int N,
                          const unsigned int block_size,
                          hipper::stream_t stream)
    {
    // clamp block size
    static unsigned int max_block_size = UINT_MAX;
    if (max_block_size == UINT_MAX)
        {
        hipper::funcAttributes_t attr;
        hipper::funcGetAttributes(&attr, reinterpret_cast<const void*>(kernel::lbvh_gen_tree));
        max_block_size = attr.maxThreadsPerBlock;
        }
    const unsigned int run_block_size = (block_size < max_block_size) ? block_size : max_block_size;
    const unsigned int num_blocks = ((N-1) + run_block_size - 1)/run_block_size;

    hipper::KernelLauncher launcher(num_blocks, run_block_size, stream);
    launcher(kernel::lbvh_gen_tree, tree, d_codes, N);
    }

//! Bubble the bounding boxes up the tree hierarchy.
/*!
 * \param tree LBVH tree (raw pointers).
 * \param d_locks Temporary storage for state of internal nodes.
 * \param insert The insert operation to obtain the aabbs
 * \param N Number of primitives.
 * \param block_size Number of CUDA threads per block.
 * \param stream CUDA stream for kernel execution.
 *
 * \tparam InsertOpT the kind of insert operation
 *
 * \sa kernel::lbvh_bubble_aabbs
 *
 * \a d_locks is overwritten before the kernel is launched.
 */
template<class InsertOpT>
void lbvh_bubble_aabbs(const LBVHData tree,
                       const InsertOpT& insert,
                       unsigned int *d_locks,
                       const unsigned int N,
                       const unsigned int block_size,
                       hipper::stream_t stream)
    {
    hipper::memsetAsync(d_locks, 0, (N-1)*sizeof(unsigned int), stream);

    // clamp block size
    static unsigned int max_block_size = UINT_MAX;
    if (max_block_size == UINT_MAX)
        {
        hipper::funcAttributes_t attr;
        hipper::funcGetAttributes(&attr, reinterpret_cast<const void*>(kernel::lbvh_bubble_aabbs<InsertOpT>));
        max_block_size = attr.maxThreadsPerBlock;
        }
    const unsigned int run_block_size = (block_size < max_block_size) ? block_size : max_block_size;
    const unsigned int num_blocks = (N + run_block_size - 1)/run_block_size;

    hipper::KernelLauncher launcher(num_blocks, run_block_size, stream);
    launcher(kernel::lbvh_bubble_aabbs<InsertOpT>, tree, insert, d_locks, N);
    }

//! Set data for a one-primitive LBVH.
/*!
 * \param tree LBVH tree (raw pointers).
 * \param insert The insert operation to obtain the ONE aabb.
 * \param stream CUDA stream for kernel execution.
 *
 * \tparam InsertOpT the kind of insert operation
 *
 * \sa kernel::lbvh_one_primitive
 */
template<class InsertOpT>
void lbvh_one_primitive(const LBVHData tree,
                        const InsertOpT& insert,
                        hipper::stream_t stream)
    {
    hipper::KernelLauncher launcher(1,1,stream);
    launcher(kernel::lbvh_one_primitive<InsertOpT>, tree, insert);
    }

} // end namespace gpu
} // end namespace neighbor

#endif // NEIGHBOR_KERNELS_LBVH_CUH_
