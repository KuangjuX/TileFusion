// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "cell/copy/constants.hpp"
#include "cell/copy/vectorize.hpp"
#include "cell/copy/warp.hpp"
#include "traits/base.hpp"
#include "types/mod.hpp"

namespace tilefusion::cell::copy {

/**
 * @brief Load a BastTile Matrix from Global memory to Register.
 * @param Global Global memory tile type.
 * @param BaseTile BaseTile type.
 * @param kType Global Layout type.
 */
template <typename Global, typename BaseTile, const tl::Layout kType>
struct GlobalToRegMatLoader;

template <typename Global_, typename BaseTile_>
struct GlobalToRegMatLoader<Global_, BaseTile_, tl::Layout::kRowMajor> {
    using Global = Global_;
    using BaseTile = BaseTile_;
    using DType = Global::DType;

    static constexpr int kStride = Global::kRowStride;

    DEVICE void operator()(const DType* src, BaseTile& dst) {
        Vectorize<DType, 2> vectorize;
        vectorize.copy(src + 0 * kStride + 0, &dst(0, 0));
        vectorize.copy(src + 0 * kStride + 8, &dst(1, 0));
        vectorize.copy(src + 8 * kStride + 0, &dst(0, 2));
        vectorize.copy(src + 8 * kStride + 8, &dst(1, 2));
    }
};

template <typename Global_, typename BaseTile_>
struct GlobalToRegMatLoader<Global_, BaseTile_, tl::Layout::kColMajor> {
    using Global = Global_;
    using BaseTile = BaseTile_;
    using DType = Global::DType;

    static constexpr int kStride = Global::kColStride;

    DEVICE void operator()(const DType* src, BaseTile& dst) {
        Vectorize<DType, 2> vectorize;
        vectorize.copy(src + 0 * kStride + 0, &dst(0, 0));
        vectorize.copy(src + 0 * kStride + 8, &dst(0, 1));
        vectorize.copy(src + 8 * kStride + 0, &dst(2, 0));
        vectorize.copy(src + 8 * kStride + 8, &dst(2, 1));
    }
};

/**
 * @brief Store a [`BaseTile`] to Global memory.
 * @param Global Global memory tile type.
 * @param BaseTile BaseTile type.
 * @param kType Global Layout type.
 */
template <typename Global_, typename BaseTile_, const tl::Layout kType>
struct RegToGlobalMatStorer;

template <typename Global_, typename BaseTile_>
struct RegToGlobalMatStorer<Global_, BaseTile_, tl::Layout::kRowMajor> {
    using Global = Global_;
    using BaseTile = BaseTile_;
    using DType = Global::DType;

    static constexpr int kStride = Global::kRowStride;

    DEVICE void operator()(const BaseTile& src, DType* dst) {
        Vectorize<DType, 2> vectorize;
        vectorize.copy(&src(0, 0), dst + 0 * kStride + 0);
        vectorize.copy(&src(1, 0), dst + 0 * kStride + 8);
        vectorize.copy(&src(0, 2), dst + 8 * kStride + 0);
        vectorize.copy(&src(1, 2), dst + 8 * kStride + 8);
    }
};

template <typename Global_, typename BaseTile_>
struct RegToGlobalMatStorer<Global_, BaseTile_, tl::Layout::kColMajor> {
    using Global = Global_;
    using BaseTile = BaseTile_;
    using DType = Global::DType;

    static constexpr int kStride = Global::kColStride;

    DEVICE void operator()(const BaseTile& src, DType* dst) {
        Vectorize<DType, 2> vectorize;
        vectorize.copy(&src(0, 0), dst + 0 * kStride + 0);
        vectorize.copy(&src(0, 1), dst + 0 * kStride + 8);
        vectorize.copy(&src(2, 0), dst + 8 * kStride + 0);
        vectorize.copy(&src(2, 1), dst + 8 * kStride + 8);
    }
};

/**
 * @brief Load a RegTile from Global memory to Register.
 * @param Global Global memory tile type.
 * @param Reg Register tile type.
 * @param kRowExec Number of times a `RegTile` is executed along the row
 * @param kColExec Number of times a `RegTile` is executed along the column
 * @param kType Global Layout type.
 */
template <typename Global, typename Reg, const int kRowExec, const int kColExec,
          const tl::Layout kType>
struct GlobalToRegLoaderImpl;

template <typename Global_, typename Reg_, const int kRowExec_,
          const int kColExec_>
struct GlobalToRegLoaderImpl<Global_, Reg_, kRowExec_, kColExec_,
                             tl::Layout::kRowMajor> {
    using Global = Global_;
    using Reg = Reg_;
    using DType = typename Global::DType;
    using BaseShape = traits::BaseTileShape<DType>;

    static constexpr int kRowExec = kRowExec_;
    static constexpr int kColExec = kColExec_;

    DEVICE void operator()(const DType* src, Reg& dst) {
        int lane_id = threadIdx.x % WARP_SIZE;
        const DType* data;

        using Loader = GlobalToRegMatLoader<Global, typename Reg::DType,
                                            tl::Layout::kRowMajor>;
        Loader loader;

#pragma unroll
        for (int i = 0; i < kRowExec; ++i) {
            int row = i * BaseShape::kCols + lane_id / 4;
#pragma unroll
            for (int j = 0; j < kColExec; ++j) {
                int col = j * BaseShape::kRows + (lane_id % 4) * 2;

                data = src + row * Global::kRowStride + col;
                loader(data, dst(i, j));
            }
        }
    }
};

template <typename Global_, typename Reg_, const int kRowExec_,
          const int kColExec_>
struct GlobalToRegLoaderImpl<Global_, Reg_, kRowExec_, kColExec_,
                             tl::Layout::kColMajor> {
    using Global = Global_;
    using Reg = Reg_;
    using DType = typename Global::DType;
    using BaseShape = traits::BaseTileShape<DType>;

    static constexpr int kRowExec = kRowExec_;
    static constexpr int kColExec = kColExec_;

    DEVICE void operator()(const DType* src, Reg& dst) {
        int lane_id = threadIdx.x % WARP_SIZE;

        const DType* data;
        using Loader = GlobalToRegMatLoader<Global, typename Reg::DType,
                                            tl::Layout::kColMajor>;
        Loader loader;

#pragma unroll
        for (int i = 0; i < kColExec; ++i) {
            int col = i * BaseShape::kRows + lane_id / 4;
            for (int j = 0; j < kRowExec; ++j) {
                int row = j * BaseShape::kCols + (lane_id % 4) * 2;
                data = src + col * Global::kColStride + row;

                loader(data, dst(j, i));
            }
        }
    }
};

/**
 * @brief Store a RegTile to Global memory.
 * @param Global Global memory tile type.
 * @param Reg Register tile type.
 * @param kRowExec Number of times a `RegTile` is executed along the row
 * @param kColExec Number of times a `RegTile` is executed along the column
 * @param kType Global Layout type.
 */
template <typename Global, typename Reg, const int kRowExec, const int kColExec,
          const tl::Layout kType>
struct RegToGlobalStorerImpl;

template <typename Global_, typename Reg_, const int kRowExec_,
          const int kColExec_>
struct RegToGlobalStorerImpl<Global_, Reg_, kRowExec_, kColExec_,
                             tl::Layout::kRowMajor> {
    using Global = Global_;
    using Reg = Reg_;
    using DType = typename Global::DType;

    using BaseShape = traits::BaseTileShape<DType>;

    static constexpr int kRowExec = kRowExec_;
    static constexpr int kColExec = kColExec_;

    DEVICE void operator()(const Reg& src, DType* dst) {
        int lane_id = threadIdx.x % WARP_SIZE;
        DType* data;

        using Storer = RegToGlobalMatStorer<Global, typename Reg::DType,
                                            tl::Layout::kRowMajor>;
        Storer storer;

#pragma unroll
        for (int i = 0; i < kRowExec; ++i) {
            int row = i * BaseShape::kCols + lane_id / 4;
#pragma unroll
            for (int j = 0; j < kColExec; ++j) {
                int col = j * BaseShape::kRows + (lane_id % 4) * 2;
                data = dst + row * Global::kRowStride + col;

                storer(src(i, j), data);
            }
        }
    }
};

template <typename Global_, typename Reg_, const int kRowExec_,
          const int kColExec_>
struct RegToGlobalStorerImpl<Global_, Reg_, kRowExec_, kColExec_,
                             tl::Layout::kColMajor> {
    using Global = Global_;
    using Reg = Reg_;
    using DType = typename Global::DType;

    using BaseShape = traits::BaseTileShape<DType>;

    static constexpr int kRowExec = kRowExec_;
    static constexpr int kColExec = kColExec_;

    DEVICE void operator()(const Reg& src, DType* dst) {
        int lane_id = threadIdx.x % WARP_SIZE;
        DType* data;

        using Storer = RegToGlobalMatStorer<Global, typename Reg::DType,
                                            tl::Layout::kColMajor>;
        Storer storer;

#pragma unroll
        for (int i = 0; i < kColExec; ++i) {
            int col = i * BaseShape::kRows + lane_id / 4;
#pragma unroll
            for (int j = 0; j < kRowExec; ++j) {
                int row = j * BaseShape::kCols + (lane_id % 4) * 2;
                data = dst + col * Global::kColStride + row;

                storer(src(j, i), data);
            }
        }
    }
};

/**
 * @brief Load a data tile from Global memory to Register based on the warp
 *        reuse mode.
 * @param Reg_ Register tile type.
 * @param WarpLayout_ Warp layout type.
 * @param kMode_ Warp reuse mode.
 */
template <typename Reg_, typename WarpLayout_, const WarpReuse kMode_>
struct GlobalToRegLoader {
    using Reg = Reg_;
    using DType = typename Reg::DType::DType;
    using WarpLayout = WarpLayout_;
    static constexpr WarpReuse kMode = kMode_;

    // how many times a `BaseTile` is executed along the row and column
    // direction.
    static constexpr int kRowExec = Reg::kRows;
    static constexpr int kColExec = Reg::kCols;

    template <typename Global>
    DEVICE void operator()(const Global& src, Reg& dst) {
        // advance the pointer to input data to the current warp
        // according to warp reuse mode.
        int offset = global_offset_.template get_warp_offset<Global>();

        using Loader = GlobalToRegLoaderImpl<Global, Reg, kRowExec, kColExec,
                                             Global::kType>;
        Loader loader;
        loader(src.data() + offset, dst);
    }

  private:
    using GlobalOffset = warp::GlobalOffsetHelper<WarpLayout, WarpReuse::kCont>;

    GlobalOffset global_offset_;
};

/**
 * @brief Store a data tile from Register to Global memory based on the warp
 *        reuse mode.
 * @param Global_ Global memory tile type.
 * @param Reg_ Register tile type.
 * @param WarpLayout_ Warp layout type.
 */
template <typename Global_, typename Reg_, typename WarpLayout_>
struct RegToGlobalStorer {
    using Global = Global_;
    using Reg = Reg_;
    using DType = typename Global::DType;
    using WarpLayout = WarpLayout_;

    // how many times a `BaseTile` is executed along the row and column
    // direction.
    static constexpr int kRowExec = Reg::kRows;
    static constexpr int kColExec = Reg::kCols;

    DEVICE void operator()(const Reg& src, Global& dst) {
        DType* dst_ptr = dst.mutable_data();

        // advance the pointer to output data to the current warp
        // according to warp reuse mode.
        int offset = global_offset_.template get_warp_offset<Global>();

        using Storer = RegToGlobalStorerImpl<Global, Reg, kRowExec, kColExec,
                                             Global::kType>;
        Storer storer;
        storer(src, dst_ptr + offset);
    }

    using GlobalOffset = warp::GlobalOffsetHelper<WarpLayout, WarpReuse::kCont>;

    GlobalOffset global_offset_;
};
}  // namespace tilefusion::cell::copy
