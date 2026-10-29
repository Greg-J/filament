/*
 * Copyright (C) 2021 The Android Open Source Project
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

#ifndef TNT_FILAMENT_BACKEND_PRIVATE_HANDLEALLOCATOR_H
#define TNT_FILAMENT_BACKEND_PRIVATE_HANDLEALLOCATOR_H

#include <backend/Handle.h>

#include <utils/Allocator.h>
#include <utils/compiler.h>
#include <utils/CString.h>
#include <utils/debug.h>
#include <utils/ImmutableCString.h>
#include <utils/Log.h>
#include <utils/ostream.h>
#include <utils/Panic.h>

#include <tsl/robin_map.h>

#include <atomic>
#include <cstddef>
#include <exception>
#include <mutex>
#include <type_traits>
#include <unordered_map>
#include <utility>

#include <stddef.h>
#include <stdint.h>

#if defined(FILAMENT_DEBUG_MUTEX)
#define HandleAllocatorGL   HandleAllocator<32,  96, 312>
#else
#define HandleAllocatorGL   HandleAllocator<32,  96, 184>    // ~4520 / pool / MiB
#endif
#define HandleAllocatorVK   HandleAllocator<64, 160, 312>    // ~1820 / pool / MiB
#define HandleAllocatorMTL  HandleAllocator<32,  64, 552>    // ~1660 / pool / MiB
// TODO WebGPU examine right size of handles
#define HandleAllocatorWGPU HandleAllocator<64, 160, 552>    // ~1820 / pool / MiB

namespace filament::backend {

// This is used to not duplicate the code for the Tags management
class DebugTag {
public:
    DebugTag();
    void writePoolHandleTag(HandleBase::HandleId key, utils::ImmutableCString&& tag) noexcept;
    void writeHeapHandleTag(HandleBase::HandleId key, utils::ImmutableCString&& tag) noexcept;
    utils::ImmutableCString findHandleTag(HandleBase::HandleId key) const noexcept;

private:
    // This is used to associate a tag to a handle. mDebugTags is only written the in the main
    // driver thread, but it can be accessed from any thread, because it's called from handle_cast<>
    // which is used by synchronous calls.
    mutable utils::Mutex mDebugTagLock;
    tsl::robin_map<HandleBase::HandleId, utils::ImmutableCString> mDebugTags UTILS_GUARDED_BY(mDebugTagLock);
};

/*
 * A utility class to efficiently allocate and manage Handle<>
 */
template<size_t P0, size_t P1, size_t P2>
class HandleAllocator : public DebugTag {
public:
    static constexpr size_t bucketSizesSum = P0 + P1 + P2;

    HandleAllocator(const char* name, size_t size);
    HandleAllocator(const char* name, size_t size,
            bool disableUseAfterFreeCheck, bool disableHeapHandleTags);
    HandleAllocator(HandleAllocator const& rhs) = delete;
    HandleAllocator& operator=(HandleAllocator const& rhs) = delete;
    ~HandleAllocator() noexcept;

    /*
     * Constructs a D object and returns a Handle<D>
     *
     * e.g.:
     *  struct ConcreteTexture : public HwTexture {
     *      ConcreteTexture(int w, int h);
     *  };
     *  Handle<ConcreteTexture> h = allocateAndConstruct(w, h);
     *
     */
    template<typename D, typename ... ARGS>
    Handle<D> allocateAndConstruct(ARGS&& ... args) {
        Handle<D> h{ allocateHandle<D>() };
        D* addr = handle_cast<D*>(h);
        new(addr) D(std::forward<ARGS>(args)...);
        return h;
    }

    /*
     * Allocates (without constructing) a D object and returns a Handle<D>
     *
     * e.g.:
     *  struct ConcreteTexture : public HwTexture {
     *      ConcreteTexture(int w, int h);
     *  };
     *  Handle<ConcreteTexture> h = allocate();
     *
     */
    template<typename D>
    Handle<D> allocate() {
        Handle<D> h{ allocateHandle<D>() };
        return h;
    }


    /*
     * Destroys the object D at Handle<B> and construct a new D in its place
     * e.g.:
     *  Handle<ConcreteTexture> h = allocateAndConstruct(w, h);
     *  ConcreteTexture* p = destroyAndConstruct(h, w, h);
     */
    template<typename D, typename B, typename ... ARGS>
    std::enable_if_t<std::is_base_of_v<B, D>, D>*
    destroyAndConstruct(Handle<B> const& handle, ARGS&& ... args) {
        assert_invariant(handle);
        D* addr = handle_cast<D*>(const_cast<Handle<B>&>(handle));
        assert_invariant(addr);
        // currently we implement construct<> with dtor+ctor, we could use operator= also
        // but all our dtors are trivial, ~D() is actually a noop.
        addr->~D();
        new(addr) D(std::forward<ARGS>(args)...);
        return addr;
    }

    /*
     * Construct a new D at Handle<B>
     * e.g.:
     *  Handle<ConcreteTexture> h = allocate();
     *  ConcreteTexture* p = construct(h, w, h);
     */
    template<typename D, typename B, typename ... ARGS>
    std::enable_if_t<std::is_base_of_v<B, D>, D>*
    construct(Handle<B> const& handle, ARGS&& ... args) noexcept {
        assert_invariant(handle);
        D* addr = handle_cast<D*>(const_cast<Handle<B>&>(handle));
        assert_invariant(addr);
        new(addr) D(std::forward<ARGS>(args)...);
        return addr;
    }

    /*
     * Destroy the object D at Handle<B> and frees Handle<B>
     * e.g.:
     *      Handle<HwTexture> h = ...;
     *      ConcreteTexture* p = handle_cast<ConcreteTexture*>(h);
     *      deallocate(h, p);
     */
    template <typename B, typename D,
            typename = std::enable_if_t<std::is_base_of_v<B, D>, D>>
    void deallocate(Handle<B>& handle, D const* p) noexcept {
        // allow to destroy the nullptr, similarly to operator delete
        if (p) {
            p->~D();
            deallocateHandle<D>(handle.getId());
        }
    }

    /*
     * Destroy the object D at Handle<B> and frees Handle<B>
     * e.g.:
     *      Handle<HwTexture> h = ...;
     *      deallocate<GLTexture>(h);
     */
    template<typename D, typename B,
            typename = std::enable_if_t<std::is_base_of_v<B, D>, D>>
    void deallocate(Handle<B>& handle) noexcept {
        D const* d = handle_cast<const D*>(handle);
        deallocate(handle, d);
    }

    /*
     * returns a D* from a Handle<B>. B must be a base of D.
     * e.g.:
     *      Handle<HwTexture> h = ...;
     *      ConcreteTexture* p = handle_cast<ConcreteTexture*>(h);
     */
    template<typename Dp, typename B>
    inline std::enable_if_t<
            std::is_pointer_v<Dp> &&
            std::is_base_of_v<B, std::remove_pointer_t<Dp>>, Dp>
    handle_cast(Handle<B>& handle) {
        assert_invariant(handle);
        return static_cast<Dp>(handleCast(handle.getId()));
    }

    utils::ImmutableCString getHandleTag(HandleBase::HandleId key) const noexcept;

    template<typename B>
    bool is_valid(Handle<B>& handle) {
        if (!handle) {
            // null handles are invalid
            return false;
        }
        auto [p, tag] = handleToPointer(handle.getId());
        if (isPoolHandle(handle.getId())) {
            uint8_t const age = (tag & HANDLE_AGE_MASK) >> HANDLE_AGE_SHIFT;
            auto const pNode = static_cast<typename Allocator::Node*>(p);
            uint8_t const expectedAge = pNode[-1].age;
            return expectedAge == age;
        }
        return p != nullptr;
    }

    template<typename Dp, typename B>
    inline std::enable_if_t<
            std::is_pointer_v<Dp> &&
            std::is_base_of_v<B, std::remove_pointer_t<Dp>>, Dp>
    handle_cast(Handle<B> const& handle) {
        return handle_cast<Dp>(const_cast<Handle<B>&>(handle));
    }

    void associateTagToHandle(HandleBase::HandleId id, utils::ImmutableCString&& tag) noexcept;

private:

    template<typename D>
    static constexpr size_t getBucketSize() noexcept {
        if constexpr (sizeof(D) <= P0) { return P0; }
        if constexpr (sizeof(D) <= P1) { return P1; }
        static_assert(sizeof(D) <= P2);
        return P2;
    }

    /*
     * Memory layout of Allocator's shared arena:
     *
     * Instead of partitioning the arena upfront into three fixed sub-pools, all three size buckets
     * (P0, P1, P2) share a single contiguous arena [mArea.begin(), mEnd) with a lazy high-water
     * mark pointer (mCur) and three recycled freelists (mPool0, mPool1, mPool2):
     *
     *   mArea.begin()                                             mCur                  mEnd
     *   v                                                         v                     v
     *   +-----+------------------+------------------+-------------+---------------------+
     *   | pad |  slot (P1)       |  slot (P0)       |  slot (P2)  |  untouched pages    |
     *   +-----+------------------+------------------+-------------+---------------------+
     *         ^                  ^                  ^             ^
     *         p0                 p1                 p2            next bump allocation
     *         |<-- touched physical memory (high-water mark) ---->|
     *
     * Per-slot layout and alignment:
     *
     * mCur is initially aligned with pointermath::align(mArea.begin(), MIN_ALIGNMENT, sizeof(Node))
     * so that p0 is MIN_ALIGNMENT-aligned and has at least sizeof(Node) (1 byte) before it.
     * Advancing mCur by getAlignedSlotSize<Pi>() (which is a multiple of MIN_ALIGNMENT) guarantees
     * that every subsequent mCur stays MIN_ALIGNMENT-aligned and reserves space at the end of the
     * slot for the next slot's Node header:
     *
     *         |<--------------- getAlignedSlotSize<Pi>() -------------->|
     *   ... --+----------------------+------------------------+---------+------+ ...
     *         |   payload (Pi bytes) |   alignment padding    |  Node   | next |
     *         |   (or FreeList next) |  (slotSize - Pi - 1 B) | { age } | slot |
     *   ... --+----------------------+------------------------+---------+------+ ...
     *   ^     ^                                               ^         ^
     *   |     p (MIN_ALIGNMENT-aligned)                       |         next p (or mCur)
     *   p[-1] = Node { age } for p                            next_p[-1] = Node { age } for next_p
     *
     * The padding is getAlignedSlotSize<Pi>() - Pi - sizeof(Node) bytes, i.e. MIN_ALIGNMENT - 1
     * when Pi is a multiple of MIN_ALIGNMENT, and fewer otherwise (e.g. 7 bytes for P2 = 184 with
     * a 16-byte MIN_ALIGNMENT, since the slot rounds up to 192).
     *
     * Allocation & deallocation flow:
     *   1. alloc(size) first checks the matching recycled freelist (mPool0, mPool1, or mPool2):
     *        mPool0 (P0): mHead -> [freed P0 slot] -> [freed P0 slot] -> nullptr
     *        mPool1 (P1): mHead -> [freed P1 slot] -> nullptr
     *        mPool2 (P2): mHead -> nullptr
     *      If non-empty, it pops a recycled slot p and returns its existing p[-1].age.
     *   2. Otherwise, it bump-allocates from mCur, advances mCur by getAlignedSlotSize<Pi>(),
     *      and initializes p[-1].age = 0. Pages beyond mCur are never touched.
     *   3. free(p, size, age) verifies p[-1].age == age, increments p[-1].age (mod 16), and
     *      pushes p onto mPool0, mPool1, or mPool2 for future reuse.
     */
    class Allocator {
        friend class HandleAllocator;
        static constexpr size_t MIN_ALIGNMENT = alignof(std::max_align_t);
        struct Node { uint8_t age; };
        // Freed slots are threaded through utils::FreeList, which stores its `next` pointer in
        // the first bytes of the payload, so even the smallest bucket must be able to hold it.
        static_assert(P0 >= sizeof(utils::FreeList::Node),
                "P0 must accommodate at least a utils::FreeList::Node");
        // Note: having a 1-byte Node structure before each allocation increases all pool slot
        // strides by MIN_ALIGNMENT bytes because of alignment restrictions.
        template<size_t SIZE>
        static constexpr size_t getAlignedSlotSize() noexcept {
            return (SIZE + sizeof(Node) + MIN_ALIGNMENT - 1) & ~(MIN_ALIGNMENT - 1);
        }
        utils::FreeList mPool0;
        utils::FreeList mPool1;
        utils::FreeList mPool2;
        char* mCur = nullptr;
        char* mEnd = nullptr;
        UTILS_UNUSED_IN_RELEASE const utils::AreaPolicy::HeapArea& mArea;
        bool mUseAfterFreeCheckDisabled;
    public:
        explicit Allocator(const utils::AreaPolicy::HeapArea& area, bool disableUseAfterFreeCheck);

        static constexpr size_t getAlignment() noexcept { return MIN_ALIGNMENT; }

        // this is in fact always called with a constexpr size argument
        [[nodiscard]] inline void* alloc(size_t size, size_t, size_t, uint8_t* outAge) noexcept {
            void* p = nullptr;
            if      (size <= P0) p = mPool0.pop(0);
            else if (size <= P1) p = mPool1.pop(0);
            else if (size <= P2) p = mPool2.pop(0);
            if (UTILS_LIKELY(p)) {
                Node const* const pNode = static_cast<Node const*>(p);
                // we are guaranteed to have at least sizeof<Node> bytes of extra storage before
                // the allocation address.
                *outAge = pNode[-1].age;
                return p;
            }

            size_t const slotSize = (size <= P0) ? getAlignedSlotSize<P0>() :
                                    (size <= P1) ? getAlignedSlotSize<P1>() :
                                                   getAlignedSlotSize<P2>();
            if (UTILS_LIKELY(size <= P2 && static_cast<size_t>(mEnd - mCur) >= slotSize)) {
                p = mCur;
                mCur += slotSize;
                Node* const pNode = static_cast<Node*>(p);
                pNode[-1].age = 0;
                *outAge = 0;
            }
            return p;
        }

        // this is in fact always called with a constexpr size argument
        inline void free(void* p, size_t size, uint8_t age) noexcept {
            assert_invariant(p >= mArea.begin() && (char*)p + size <= mCur);

            // check for double-free
            Node* const pNode = static_cast<Node*>(p);
            uint8_t& expectedAge = pNode[-1].age;
            if (UTILS_UNLIKELY(!mUseAfterFreeCheckDisabled)) {
                FILAMENT_CHECK_POSTCONDITION(expectedAge == age) <<
                        "double-free of Handle of size " << size << " at " << p;
            }
            expectedAge = (expectedAge + 1) & 0xF; // fixme

            if (size <= P0) { mPool0.push(p); return; }
            if (size <= P1) { mPool1.push(p); return; }
            if (size <= P2) { mPool2.push(p); return; }
        }
    };

// FIXME: We should be using a Spinlock here, at least on platforms where mutexes are not
//        efficient (i.e. non-Linux). However, we've seen some hangs on that spinlock, which
//        we don't understand well (b/308029108).
#ifndef NDEBUG
    using HandleArena = utils::Arena<Allocator,
            utils::LockingPolicy::Mutex,
            utils::TrackingPolicy::DebugAndHighWatermark>;
#else
    using HandleArena = utils::Arena<Allocator,
            utils::LockingPolicy::Mutex>;
#endif

    // allocateHandle()/deallocateHandle() selects the pool to use at compile-time based on the
    // allocation size this is always inlined, because all these do is to call
    // allocateHandleInPool()/deallocateHandleFromPool() with the right pool size.
    template<typename D>
    HandleBase::HandleId allocateHandle() {
        constexpr size_t BUCKET_SIZE = getBucketSize<D>();
        return allocateHandleInPool<BUCKET_SIZE>();
    }

    template<typename D>
    void deallocateHandle(HandleBase::HandleId id) noexcept {
        constexpr size_t BUCKET_SIZE = getBucketSize<D>();
        deallocateHandleFromPool<BUCKET_SIZE>(id);
    }

    // allocateHandleInPool()/deallocateHandleFromPool() is NOT inlined, which will cause three
    // versions to be generated, one for each pool. Because the arena is synchronized,
    // the code generated is not trivial (even if it's not insane either).
    template<size_t SIZE>
    UTILS_NOINLINE
    HandleBase::HandleId allocateHandleInPool() {
        uint8_t age;
        void* p = mHandleArena.alloc(SIZE, alignof(std::max_align_t), 0, &age);
        if (UTILS_LIKELY(p)) {
            uint32_t const tag = (uint32_t(age) << HANDLE_AGE_SHIFT) & HANDLE_AGE_MASK;
            return arenaPointerToHandle(p, tag);
        }
        return allocateHandleSlow(SIZE);
    }

    template<size_t SIZE>
    UTILS_NOINLINE
    void deallocateHandleFromPool(HandleBase::HandleId id) noexcept {
        if (UTILS_LIKELY(isPoolHandle(id))) {
            auto [p, tag] = handleToPointer(id);
            uint8_t const age = (tag & HANDLE_AGE_MASK) >> HANDLE_AGE_SHIFT;
            mHandleArena.free(p, SIZE, age);
        } else {
            deallocateHandleSlow(id, SIZE);
        }
    }

    // number if bits allotted to the handle's age (currently 4 max)
    static constexpr uint32_t HANDLE_AGE_BIT_COUNT = 4;
    // number if bits allotted to the handle's debug tag (HANDLE_AGE_BIT_COUNT max)
    static constexpr uint32_t HANDLE_DEBUG_TAG_BIT_COUNT = 2;
    // bit shift for both the age and debug tag
    static constexpr uint32_t HANDLE_AGE_SHIFT = 27;
    // mask for the heap (vs pool) flag
    static constexpr uint32_t HANDLE_HEAP_FLAG = 0x80000000u;
    // mask for the age
    static constexpr uint32_t HANDLE_AGE_MASK =
            ((1 << HANDLE_AGE_BIT_COUNT) - 1) << HANDLE_AGE_SHIFT;
    // mask for the debug tag
    static constexpr uint32_t HANDLE_DEBUG_TAG_MASK =
            ((1 << HANDLE_DEBUG_TAG_BIT_COUNT) - 1) << HANDLE_AGE_SHIFT;
    // mask for the index
    static constexpr uint32_t HANDLE_INDEX_MASK = 0x07FFFFFFu;

    static_assert(HANDLE_DEBUG_TAG_BIT_COUNT <= HANDLE_AGE_BIT_COUNT);

    static bool isPoolHandle(HandleBase::HandleId id) noexcept {
        return (id & HANDLE_HEAP_FLAG) == 0u;
    }

    HandleBase::HandleId allocateHandleSlow(size_t size);
    void deallocateHandleSlow(HandleBase::HandleId id, size_t size) noexcept;

    // We inline this because it's just 4 instructions in the fast case
   std::pair<void*, uint32_t> handleToPointer(HandleBase::HandleId id) const noexcept {
        // note: the null handle will end-up returning nullptr b/c it'll be handled as
        // a non-pool handle.
        if (UTILS_LIKELY(isPoolHandle(id))) {
            char* const base = (char*)mHandleArena.getArea().begin();
            uint32_t const tag = id & HANDLE_AGE_MASK;
            size_t const offset = (id & HANDLE_INDEX_MASK) * Allocator::getAlignment();
            return { static_cast<void*>(base + offset), tag };
        }
        return { handleToPointerSlow(id), 0 };
    }

    void* handleToPointerSlow(HandleBase::HandleId id) const noexcept;
    void* handleCast(HandleBase::HandleId id) const;

    // We inline this because it's just 3 instructions
   HandleBase::HandleId arenaPointerToHandle(void* p, uint32_t tag) const noexcept {
        char* const base = (char*)mHandleArena.getArea().begin();
        size_t const offset = (char*)p - base;
        assert_invariant((offset % Allocator::getAlignment()) == 0);
        auto id = HandleBase::HandleId(offset / Allocator::getAlignment());
        id |= tag & HANDLE_AGE_MASK;
        assert_invariant((id & HANDLE_HEAP_FLAG) == 0);
        return id;
    }

    HandleArena mHandleArena;

    // Below is only used when running out of space in the HandleArena
    mutable utils::Mutex mLock;
    tsl::robin_map<HandleBase::HandleId, void*> mOverflowMap UTILS_GUARDED_BY(mLock);
    std::atomic<HandleBase::HandleId> mId = 0;

    // constants
    const bool mUseAfterFreeCheckDisabled;
    const bool mHeapHandleTagsDisabled;
};

} // namespace filament::backend

#endif // TNT_FILAMENT_BACKEND_PRIVATE_HANDLEALLOCATOR_H
