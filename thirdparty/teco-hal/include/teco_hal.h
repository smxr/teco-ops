#ifndef TECO_HAL_INCLUDE_TECO_HAL_H
#define TECO_HAL_INCLUDE_TECO_HAL_H

#include "teco_hal_config.h"
#include "teco_hal_dma.hpp"
#include "teco_hal_matmul.hpp"
#include "teco_hal_queue.hpp"
#include "teco_hal_rma.hpp"

// TECOHAL_VERSION_MAJOR
#define TECOHAL_MAJOR 0
// TECOHAL_VERSION_MINOR
#define TECOHAL_MINOR 0
// TECOHAL_VERSION_PATCH
#define TECOHAL_PATCHLEVEL 1
// TECOHAL_VERSION_TYPE
#define TECOHAL_VERSION_TYPE 2
// TECOHAL_VERSION_TYPE_NO
#define TECOHAL_VERSION_TYPE_NO 0

#define TECOHAL_VERSION (TECOHAL_MAJOR * 10000 + TECOHAL_MINOR * 100 + TECOHAL_PATCHLEVEL)
namespace teco {
namespace hal {

static size_t tecoHalGetVersion(void) {
    size_t version = 0;
    version = TECOHAL_MAJOR * 100000000 + TECOHAL_MINOR * 1000000 + TECOHAL_PATCHLEVEL * 10000 +
              TECOHAL_VERSION_TYPE * 100 + TECOHAL_VERSION_TYPE_NO;
    return version;
}


}  // namespace hal
}  // namespace teco

#endif  // TECO_HAL_INCLUDE_TECO_HAL_H
