#ifndef TECO_HAL_INCLUDE_TECO_HAL_QUEUE_HPP
#define TECO_HAL_INCLUDE_TECO_HAL_QUEUE_HPP

#include "teco_hal_config.h"

namespace teco {
namespace hal {

/**
 * @brief 设备端定长环形队列 (FIFO)
 * @tparam T       数据类型
 * @tparam CAPACITY 队列最大容量
 */
template <typename T, int CAPACITY = 50>
class DeviceQueue {
 private:
    T data_[CAPACITY];
    int head_ = 0;
    int tail_ = 0;
    int size_ = 0;

 public:
    // 入队
    HAL_DEVICE void push(const T &val) {
        if (size_ < CAPACITY) {
            data_[tail_] = val;
            tail_ = (tail_ + 1) % CAPACITY;
            size_++;
        }
    }

    // 出队
    HAL_DEVICE void pop() {
        if (size_ > 0) {
            head_ = (head_ + 1) % CAPACITY;
            size_--;
        }
    }

    // 获取队首元素
    HAL_DEVICE T &front() { return data_[head_]; }
    HAL_DEVICE const T &front() const { return data_[head_]; }

    // 获取队列大小
    HAL_DEVICE int size() const { return size_; }

    // 判空
    HAL_DEVICE bool empty() const { return size_ == 0; }
};

}  // namespace hal
}  // namespace teco

#endif  // TECO_HAL_INCLUDE_TECO_HAL_QUEUE_HPP
