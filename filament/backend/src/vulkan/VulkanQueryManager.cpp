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

#include "VulkanQueryManager.h"

#include "VulkanAsyncHandles.h"
#include "VulkanCommands.h"
#include "VulkanConstants.h"
#include "VulkanContext.h"

#include <bluevk/BlueVK.h>

#include <utils/debug.h>
#include <utils/Mutex.h>
#include <utils/Panic.h>

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace filament::backend {

using namespace bluevk;

VulkanQueryManager::VulkanQueryManager(VulkanContext const& context, VkDevice device)
        : mDevice(device),
          mHostQueryReset(context.isHostQueryResetSupported() && vkResetQueryPoolEXT != nullptr) {
    static_assert(decltype(mUsed)::BIT_COUNT == MAX_TIMERS,
            "mUsed must have exactly one bit per timer slot");

    // Create a timestamp pool large enough to hold a pair of queries for each timer.
    VkQueryPoolCreateInfo const tqpCreateInfo = {
        .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
        .queryType = VK_QUERY_TYPE_TIMESTAMP,
        .queryCount = QUERY_COUNT,
    };
    VkResult const result = vkCreateQueryPool(mDevice, &tqpCreateInfo, VKALLOC, &mPool);
    FILAMENT_CHECK_POSTCONDITION(result == VK_SUCCESS) << "vkCreateQueryPool failed."
                                                       << " error=" << static_cast<int32_t>(result);

    if (mHostQueryReset) {
        vkResetQueryPoolEXT(mDevice, mPool, 0, QUERY_COUNT);
    }
}

VulkanQueryManager::~VulkanQueryManager() = default;

fvkmemory::resource_ptr<VulkanTimerQuery> VulkanQueryManager::getNextQuery(
        fvkmemory::ResourceManager* resourceManager) {
    uint32_t startingIndex = 0;
    uint32_t stoppingIndex = 0;
    {
        utils::LockGuard const lock(mMutex);
        auto const unused = ~mUsed;
        if (unused.empty()) {
            FVK_LOGE << "More than " << mUsed.size() << " timers are not supported.";
            return {};
        }

        size_t const firstUnused = unused.firstSetBit();
        mUsed.set(firstUnused);
        assert_invariant(!mQueryFences[firstUnused].starting &&
                !mQueryFences[firstUnused].stopping);
        startingIndex = static_cast<uint32_t>(firstUnused * 2);
        stoppingIndex = startingIndex + 1;
    }
    return fvkmemory::resource_ptr<VulkanTimerQuery>::construct(resourceManager, startingIndex,
            stoppingIndex);
}

void VulkanQueryManager::clearQuery(fvkmemory::resource_ptr<VulkanTimerQuery> query) {
    uint32_t const slot = query->getStartingQueryIndex() / 2;
    assert_invariant(slot < MAX_TIMERS);
    mActive.unset(slot);
    QueryFences oldFences;
    {
        utils::LockGuard const lock(mMutex);
        oldFences = std::move(mQueryFences[slot]);
        mUsed.unset(slot);
    }
}

void VulkanQueryManager::trackInFlightFence(uint32_t slot, std::shared_ptr<VulkanCmdFence> fence) {
    auto& fences = mInFlightFences[slot];
    if (fences.empty() || fences.back() != fence) {
        fences.push_back(std::move(fence));
    }
}

void VulkanQueryManager::beginQuery(VulkanCommandBuffer const* commands,
        fvkmemory::resource_ptr<VulkanTimerQuery> query) {
    uint32_t const index = query->getStartingQueryIndex();
    uint32_t const slot = index / 2;
    assert_invariant(slot < MAX_TIMERS);
    auto const cmdbuffer = commands->buffer();
    auto startingFence = commands->getFenceStatus();

    bool resetOnDevice = !mHostQueryReset;
    QueryFences oldFences;

    if (mHostQueryReset) {
        bool const needsReset = mNeedsReset.test(slot);
        bool canResetOnHost = false;
        if (needsReset) {
            auto& fences = mInFlightFences[slot];
            std::erase_if(fences,
                    [device = mDevice](std::shared_ptr<VulkanCmdFence> const& fence) {
                        VkResult status = fence->getStatus();
                        if (status == VK_NOT_READY) {
                            fence->refreshStatus(device);
                            status = fence->getStatus();
                        }
                        return status == VK_SUCCESS;
                    });
            // VUID-vkResetQueryPool-firstQuery-02741 requires all submitted commands referencing
            // these queries to have completed execution before resetting on the host. If any
            // previous submission is still in flight (or still recording), fall back to device
            // reset in the command buffer.
            canResetOnHost = fences.empty();
        }

        {
            // VUID-vkResetQueryPool-firstQuery-02742 requires that the queries being reset are
            // not in use by vkGetQueryPoolResults in other threads. getResult() calls it on the
            // main thread under mMutex, so the host reset must also be done under mMutex.
            utils::LockGuard const lock(mMutex);
            oldFences = std::exchange(mQueryFences[slot], { startingFence, nullptr });
            if (needsReset && canResetOnHost) {
                vkResetQueryPoolEXT(mDevice, mPool, index, 2);
            }
        }

        resetOnDevice = needsReset && !canResetOnHost;
    } else {
        utils::LockGuard const lock(mMutex);
        oldFences = std::exchange(mQueryFences[slot], { startingFence, nullptr });
    }

    if (resetOnDevice) {
        vkCmdResetQueryPool(cmdbuffer, mPool, index, 2);
    }
    vkCmdWriteTimestamp(cmdbuffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, mPool, index);
    mActive.set(slot);

    if (mHostQueryReset) {
        mNeedsReset.set(slot);
        trackInFlightFence(slot, std::move(startingFence));
    }
}

void VulkanQueryManager::endQuery(VulkanCommandBuffer const* commands,
        fvkmemory::resource_ptr<VulkanTimerQuery> query) {
    uint32_t const index = query->getStoppingQueryIndex();
    uint32_t const slot = query->getStartingQueryIndex() / 2;
    assert_invariant(slot < MAX_TIMERS);
    if (!mActive.test(slot)) {
        FVK_LOGW << "endTimerQuery called without a matching beginTimerQuery.";
        return;
    }
    mActive.unset(slot);

    vkCmdWriteTimestamp(commands->buffer(), VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, mPool, index);

    auto stoppingFence = commands->getFenceStatus();
    if (mHostQueryReset) {
        mNeedsReset.set(slot);
        trackInFlightFence(slot, stoppingFence);
    }

    std::shared_ptr<VulkanCmdFence> oldStoppingFence;
    {
        utils::LockGuard const lock(mMutex);
        assert_invariant(mQueryFences[slot].starting != nullptr);
        oldStoppingFence = std::exchange(mQueryFences[slot].stopping, std::move(stoppingFence));
    }
}

VulkanQueryManager::QueryResult VulkanQueryManager::getResult(
        fvkmemory::resource_ptr<VulkanTimerQuery> query) {
    uint32_t const index = query->getStartingQueryIndex();
    uint32_t const slot = index / 2;
    assert_invariant(slot < MAX_TIMERS);

    utils::LockGuard const lock(mMutex);
    // getTimerQueryValue is a synchronous call and might occur before both timestamp commands have
    // been processed, which is an error according to the validation layer that ships in the
    // Android NDK even when VK_QUERY_RESULT_WITH_AVAILABILITY_BIT is set. Notably,
    // FRenderer::endFrame commits the swapchain before calling endTimerQuery, splitting beginQuery
    // and endQuery across two command buffers without a semaphore link between them. Both fences
    // must therefore reach VK_SUCCESS before vkGetQueryPoolResults is called. Holding mMutex across
    // both the completion check and vkGetQueryPoolResults also prevents a concurrent host reset on
    // the backend thread (VUID-vkResetQueryPool-firstQuery-02742).
    auto const& fences = mQueryFences[slot];
    if (!fences.starting || !fences.stopping ||
            fences.starting->getStatus() != VK_SUCCESS ||
            (fences.stopping != fences.starting &&
                    fences.stopping->getStatus() != VK_SUCCESS)) {
        return {};
    }

    QueryResult result;
    size_t const dataSize = sizeof(result);
    VkDeviceSize const stride = sizeof(uint64_t) * 2;
    VkResult const vkresult =
            vkGetQueryPoolResults(mDevice, mPool, index, 2, dataSize, (void*) &result, stride,
                    VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT);
    FILAMENT_CHECK_POSTCONDITION(vkresult == VK_SUCCESS || vkresult == VK_NOT_READY)
            << "vkGetQueryPoolResults error=" << static_cast<int32_t>(vkresult);
    if (vkresult == VK_NOT_READY) {
        return {};
    }
    return result;
}

void VulkanQueryManager::terminate() noexcept {
    std::array<QueryFences, MAX_TIMERS> queryFences;
    {
        utils::LockGuard const lock(mMutex);
        queryFences = std::move(mQueryFences);
        mUsed.reset();
    }
    queryFences = {};
    mInFlightFences = {};
    mNeedsReset.reset();
    mActive.reset();
    vkDestroyQueryPool(mDevice, mPool, VKALLOC);
    mPool = VK_NULL_HANDLE;
}

} // namespace filament::backend
