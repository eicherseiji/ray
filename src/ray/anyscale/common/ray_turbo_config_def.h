// Copyright 2025 The Ray Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// This header file contains configuration variables for proprietary
// RayTurbo features or for more performant default values.

// NOTE: This file should NOT be included in any file other than ray_config.h.

// TODO(irabbani): Delete this config variable and turn it on by default after
// piloting it with a few customers. See https://github.com/anyscale/rayturbo/issues/1805.
RAY_CONFIG(bool, experimental_object_manager_enable_multiple_connections, false)

// TODO(dayshah): Delete this config variable after it's on for a while. See
// https://github.com/anyscale/rayturbo/issues/2090
RAY_CONFIG(bool, lazy_subscribe_core_workers, true)
