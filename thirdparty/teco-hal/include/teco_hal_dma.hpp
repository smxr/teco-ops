#ifndef TECO_HAL_INCLUDE_TECO_HAL_DMA_HPP
#define TECO_HAL_INCLUDE_TECO_HAL_DMA_HPP

#ifndef __KAI_ARCH__
#define __KAI_ARCH__
#endif

#include "teco_hal_config.h"
#include "teco_hal_queue.hpp"  // 引入公共队列
#include "teco_hal_tensor.hpp"
#include "teco_hal_tile.hpp"

namespace teco {
namespace hal {

class DmaQueue {
 public:
    /**
     * @brief 从全局内存获取数据到本地内存 (DMA GET)
     * @note 只有声明，实现在 dma.scpp 中
     */
    template <typename T, SplitMode splitMode, std::size_t Dims>
    HAL_DEVICE void dma_get(Tensor<T, Dims> &dst, Tensor<T, Dims> &src,
                            const Tile<splitMode, Dims> &tile);

    /**
     * @brief 将本地内存数据放回全局内存 (DMA PUT)
     * @note 只有声明，实现在 dma.scpp 中
     */
    template <typename T, SplitMode splitMode, std::size_t Dims>
    HAL_DEVICE void dma_put(Tensor<T, Dims> &dst, Tensor<T, Dims> &src,
                            const Tile<splitMode, Dims> &tile);

    /**
     * @brief 等待get请求完成
     */
    HAL_DEVICE void wait_get(int num = 1);

     /**
     * @brief 等待put请求完成
     */
    HAL_DEVICE void wait_put(int num = 1);

    /**
     * @brief 获取DMA队列长度
     */
    HAL_DEVICE int getQSize();

 private:
    static constexpr int QUEUE_SIZE = 20;
    volatile int dma_reply_[QUEUE_SIZE];
    int reply_idx_ = 0;
    DeviceQueue<int> dmaGetRplyQ_;
    DeviceQueue<int> dmaPutRplyQ_;
};

}  // namespace hal
}  // namespace teco

#endif  // TECO_HAL_INCLUDE_TECO_HAL_DMA_HPP
