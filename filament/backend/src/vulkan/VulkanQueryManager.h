/*
 * Copyright (C) 2025 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef TNT_FILAMENT_BACKEND_VULKANQUERYMANAGER_H
#define TNT_FILAMENT_BACKEND_VULKANQUERYMANAGER_H

#include "vulkan/memory/ResourcePointer.h"

#include <bluevk/BlueVK.h>

#include <utils/bitset.h>
#include <utils/compiler.h>
#include <utils/Mutex.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace filament::backend {

struct VulkanCmdFence;
struct VulkanCommandBuffer;
struct VulkanContext;
struct VulkanTimerQuery;

class VulkanQueryManager {
public:
    struct QueryResult {
        uint64_t beginTime = 0;
        uint64_t beginAvailable = 0;
        uint64_t endTime = 0;
        uint64_t endAvailable = 0;
    };

    VulkanQueryManager(VulkanContext const& context, VkDevice device);
    ~VulkanQueryManager();

    // Not copy-able.
    VulkanQueryManager(VulkanQueryManager const&) = delete;
    VulkanQueryManager& operator=(VulkanQueryManager const&) = delete;

    fvkmemory::resource_ptr<VulkanTimerQuery>
            getNextQuery(fvkmemory::ResourceManager* resourceManager);
    void clearQuery(fvkmemory::resource_ptr<VulkanTimerQuery> query);
    void beginQuery(VulkanCommandBuffer const* commands,
            fvkmemory::resource_ptr<VulkanTimerQuery> query);
    void endQuery(VulkanCommandBuffer const* commands,
            fvkmemory::resource_ptr<VulkanTimerQuery> query);
    QueryResult getResult(fvkmemory::resource_ptr<VulkanTimerQuery> query);

    void terminate() noexcept;

private:
    static constexpr size_t MAX_TIMERS = 32;
    static constexpr uint32_t QUERY_COUNT = MAX_TIMERS * 2;

    struct QueryFences {
        std::shared_ptr<VulkanCmdFence> starting;
        std::shared_ptr<VulkanCmdFence> stopping;
    };

    void trackInFlightFence(uint32_t slot, std::shared_ptr<VulkanCmdFence> fence);

    VkDevice mDevice;
    VkQueryPool mPool = VK_NULL_HANDLE;
    bool const mHostQueryReset;

    // Besides guarding the members below, mMutex serializes vkResetQueryPoolEXT (backend thread)
    // against vkGetQueryPoolResults (main thread), as required by
    // VUID-vkResetQueryPool-firstQuery-02742.
    utils::Mutex mMutex;
    utils::bitset32 mUsed UTILS_GUARDED_BY(mMutex);
    // Tracks the completion fences of beginQuery and endQuery for each slot; read by getResult() on
    // the main thread under mMutex, and modified only on the backend thread under mMutex.
    std::array<QueryFences, MAX_TIMERS> mQueryFences UTILS_GUARDED_BY(mMutex);

    // Accessed exclusively on the backend thread to track open queries and verify that all command
    // buffers referencing a slot have completed before host reset.
    utils::bitset32 mActive;
    utils::bitset32 mNeedsReset;
    std::array<std::vector<std::shared_ptr<VulkanCmdFence>>, MAX_TIMERS> mInFlightFences;
};

} // namespace filament::backend

#endif // TNT_FILAMENT_BACKEND_VULKANQUERYMANAGER_H
