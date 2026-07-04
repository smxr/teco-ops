#ifndef TECO_HAL_INCLUDE_TECO_HAL_MATMUL_HPP
#define TECO_HAL_INCLUDE_TECO_HAL_MATMUL_HPP

#include "teco_hal_config.h"
#include "teco_hal_queue.hpp"
#include "teco_hal_tensor.hpp"

namespace teco {
namespace hal {

enum class MatmulDataLayout {
    DataBkn32Block,  // B:[k, n] -> B:[n/32, k, 32]，k%32==0 && n%32==0
};

/// 矩阵乘法运行时配置（可扩展）
struct MatmulConfig {
    bool trans_a = false;  // 是否转置 A（预留，当前未实现）
    bool trans_b = false;  // 是否转置 B（预留，当前未实现）
};

class Matmul {
 public:
    HAL_DEVICE Matmul(const MatmulConfig &cfg = MatmulConfig{});
    HAL_DEVICE ~Matmul();

    HAL_DEVICE void set_config(const MatmulConfig &cfg);
    HAL_DEVICE const MatmulConfig &get_config() const;

    // 计算接口（仅累积，不写回）
    template <int M = 128, int N = 32, MatmulDataLayout Layout = MatmulDataLayout::DataBkn32Block,
          typename InType = half, std::size_t Dims = 2>
    HAL_DEVICE void compute_async(const Tensor<InType, Dims> &A, const Tensor<InType, Dims> &B,
                                  bool is_lastk);

    template <typename OutType, std::size_t Dims>
    HAL_DEVICE void storeC_spm_async(Tensor<OutType, Dims> &C);

    HAL_DEVICE void wait_storeC(int num);
    HAL_DEVICE void wait_loadA();
    HAL_DEVICE void wait_loadB();

 private:
    static constexpr unsigned int QSIZE = 20;
    volatile unsigned int reply_[QSIZE];
    DeviceQueue<unsigned int, QSIZE> replyQ_;
    unsigned int reply_idx_ = 0;
    unsigned int loadA_counter_ = 0;
    unsigned int loadB_counter_ = 0;
    MatmulConfig config_;
};

}  // namespace hal
}  // namespace teco

#endif  // TECO_HAL_INCLUDE_TECO_HAL_MATMUL_HPP