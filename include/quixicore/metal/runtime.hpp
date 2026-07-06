#pragma once

#include "quixicore/metal/backend.hpp"

namespace quixicore::metal {

struct RuntimeOptions {
  bool prefer_low_latency = false;
};

} // namespace quixicore::metal
