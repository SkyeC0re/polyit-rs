//! This module provides storage solution abstractions. This is done via the [`StorageProvider`](trait@StorageProvider)
//! and [`Storage`](trait@Storage) traits. Conceptually the storage type represent the specific instance of storage in question,
//! e.g. a `Vec<f64>`, whereas the storage provider represents the underlying container or mechanism, e.g. `Vec` and provides
//! a means to create different instances using that container or mechanism type.
//!

/// A trait for represting the underlying storage container or mechanism family.
pub trait StorageProvider<T> {
    /// The storage type that this family provides for `T` type elements.
    type StorageType: Storage<T>;

    /// Create a new storage provider.
    fn new() -> Self;

    /// Create a new storage instance.
    fn new_storage(&mut self) -> Self::StorageType;

    /// Create a new storage instance with at least `capacity` capacity.
    /// Is allowed to panic if the storage type cannot support the requested capacity.
    fn storage_with_capacity(&mut self, capacity: usize) -> Self::StorageType;
}

/// Represents a slice-like storage type for a specific type of element.
pub trait Storage<T>: Clone {
    /// A reference to the underlying storage provider family.
    type Provider: StorageProvider<T, StorageType = Self>;

    /// Clears all data in the storage.
    fn clear(&mut self);

    /// Push an element to storage.
    fn push(&mut self, value: T);

    /// Pop the last element from storage.
    fn pop(&mut self) -> Option<T>;

    /// Return an immutable refrence to the data.
    fn as_slice(&self) -> &[T];

    /// Return a mutable reference to the data.
    fn as_mut_slice(&mut self) -> &mut [T];

    /// Return the length of the data. Should always concide with the
    /// the slice's length.
    #[inline]
    fn len(&self) -> usize {
        self.as_slice().len()
    }

    /// Return the capacity of the storage type.
    fn capacity(&self) -> usize;
}

#[cfg(feature = "alloc")]
pub mod vec {
    //! The `Vec` storage family.
    extern crate alloc;
    use alloc::vec::Vec;

    use super::{Storage, StorageProvider};

    /// Represents the vector storage family.
    pub struct VecStorage;

    impl<T> StorageProvider<T> for VecStorage
    where
        T: Clone,
    {
        type StorageType = Vec<T>;
        #[inline]
        fn new() -> Self {
            Self
        }

        #[inline]
        fn new_storage(&mut self) -> Self::StorageType {
            Self::StorageType::new()
        }

        #[inline]
        fn storage_with_capacity(&mut self, capacity: usize) -> Self::StorageType {
            Self::StorageType::with_capacity(capacity)
        }
    }

    impl<T> Storage<T> for Vec<T>
    where
        T: Clone,
    {
        type Provider = VecStorage;

        #[inline]
        fn clear(&mut self) {
            self.clear();
        }

        #[inline]
        fn push(&mut self, value: T) {
            self.push(value);
        }

        #[inline]
        fn pop(&mut self) -> Option<T> {
            self.pop()
        }

        #[inline]
        fn len(&self) -> usize {
            self.len()
        }

        #[inline]
        fn capacity(&self) -> usize {
            self.capacity()
        }

        #[inline]
        fn as_slice(&self) -> &[T] {
            self.as_slice()
        }

        #[inline]
        fn as_mut_slice(&mut self) -> &mut [T] {
            self.as_mut_slice()
        }
    }
}

#[cfg(feature = "tinyvec")]
pub mod tinyvec {
    //! [TinyVec](https://docs.rs/tinyvec/latest/tinyvec/) storage families.
    use super::{Storage, StorageProvider};
    #[cfg(feature = "alloc")]
    use alloc::vec::Vec;
    #[cfg(feature = "alloc")]
    pub use tinyvec::TinyVec;
    pub use tinyvec::{Array, ArrayVec};

    /// Represents the `tinyvec::Array` storage family of a particular capacity.
    pub struct TinyVecArrayStorage<const CAP: usize>;

    impl<const CAP: usize, T> StorageProvider<T> for TinyVecArrayStorage<CAP>
    where
        T: Clone,
        [T; CAP]: Array<Item = T>,
    {
        type StorageType = ArrayVec<[T; CAP]>;
        #[inline]
        fn new() -> Self {
            Self
        }

        #[inline]
        fn new_storage(&mut self) -> Self::StorageType {
            Self::StorageType::new()
        }

        #[inline]
        fn storage_with_capacity(&mut self, capacity: usize) -> Self::StorageType {
            if capacity > <[T; CAP] as Array>::CAPACITY {
                panic!(
                    "Requested capacity of {} exceeds maximum of {}",
                    capacity,
                    <[T; CAP] as Array>::CAPACITY
                )
            }
            Self::StorageType::new()
        }
    }

    impl<const CAP: usize, T> Storage<T> for ArrayVec<[T; CAP]>
    where
        T: Clone,
        [T; CAP]: Array<Item = T>,
    {
        type Provider = TinyVecArrayStorage<CAP>;

        #[inline]
        fn clear(&mut self) {
            self.clear();
        }

        #[inline]
        fn push(&mut self, value: T) {
            self.push(value);
        }

        #[inline]
        fn pop(&mut self) -> Option<T> {
            self.pop()
        }

        #[inline]
        fn len(&self) -> usize {
            self.len()
        }

        #[inline]
        fn capacity(&self) -> usize {
            self.capacity()
        }

        #[inline]
        fn as_slice(&self) -> &[T] {
            self.as_slice()
        }

        #[inline]
        fn as_mut_slice(&mut self) -> &mut [T] {
            self.as_mut_slice()
        }
    }

    #[cfg(feature = "alloc")]
    /// Represents the `tinyvec::TinyVec` storage family of a particular stack capacity.
    pub struct TinyVecStorage<const CAP: usize>;

    #[cfg(feature = "alloc")]
    impl<const CAP: usize, T> StorageProvider<T> for TinyVecStorage<CAP>
    where
        T: Clone,
        [T; CAP]: Array<Item = T>,
    {
        type StorageType = TinyVec<[T; CAP]>;
        #[inline]
        fn new() -> Self {
            Self
        }

        #[inline]
        fn new_storage(&mut self) -> Self::StorageType {
            Self::StorageType::new()
        }

        #[inline]
        fn storage_with_capacity(&mut self, capacity: usize) -> Self::StorageType {
            if capacity <= <[T; CAP] as Array>::CAPACITY {
                TinyVec::Inline(ArrayVec::new())
            } else {
                TinyVec::Heap(Vec::with_capacity(capacity))
            }
        }
    }

    #[cfg(feature = "alloc")]
    impl<const CAP: usize, T> Storage<T> for TinyVec<[T; CAP]>
    where
        T: Clone,
        [T; CAP]: Array<Item = T>,
    {
        type Provider = TinyVecStorage<CAP>;

        #[inline]
        fn clear(&mut self) {
            self.clear();
        }

        #[inline]
        fn push(&mut self, value: T) {
            self.push(value);
        }

        #[inline]
        fn pop(&mut self) -> Option<T> {
            self.pop()
        }

        #[inline]
        fn len(&self) -> usize {
            self.len()
        }

        #[inline]
        fn capacity(&self) -> usize {
            self.capacity()
        }

        #[inline]
        fn as_slice(&self) -> &[T] {
            self.as_slice()
        }

        #[inline]
        fn as_mut_slice(&mut self) -> &mut [T] {
            self.as_mut_slice()
        }
    }
}

#[cfg(feature = "arrayvec")]
pub mod arrayvec {
    //! [ArrayVec](https://docs.rs/arrayvec/latest/arrayvec/) storage family.
    use super::{Storage, StorageProvider};
    pub use arrayvec::ArrayVec;

    /// Represents the `arrayvec::ArrayVec` storage family of a particular capacity.
    pub struct ArrayVecStorage<const CAP: usize>;

    impl<const CAP: usize, T> StorageProvider<T> for ArrayVecStorage<CAP>
    where
        T: Clone,
    {
        type StorageType = ArrayVec<T, CAP>;
        #[inline]
        fn new() -> Self {
            Self
        }

        #[inline]
        fn new_storage(&mut self) -> Self::StorageType {
            Self::StorageType::new()
        }

        #[inline]
        fn storage_with_capacity(&mut self, capacity: usize) -> Self::StorageType {
            if capacity > CAP {
                panic!(
                    "Requested capacity of {} exceeds maximum of {}",
                    capacity, CAP
                )
            }
            Self::StorageType::new()
        }
    }

    impl<const CAP: usize, T> Storage<T> for ArrayVec<T, CAP>
    where
        T: Clone,
    {
        type Provider = ArrayVecStorage<CAP>;

        #[inline]
        fn clear(&mut self) {
            self.clear();
        }

        #[inline]
        fn push(&mut self, value: T) {
            self.push(value);
        }

        #[inline]
        fn pop(&mut self) -> Option<T> {
            self.pop()
        }

        #[inline]
        fn len(&self) -> usize {
            self.len()
        }

        #[inline]
        fn capacity(&self) -> usize {
            self.capacity()
        }

        #[inline]
        fn as_slice(&self) -> &[T] {
            self.as_slice()
        }

        #[inline]
        fn as_mut_slice(&mut self) -> &mut [T] {
            self.as_mut_slice()
        }
    }
}

#[cfg(feature = "smallvec")]
pub mod smallvec {
    //! [SmallVec](https://docs.rs/smallvec/latest/smallvec/) storage family.
    use super::{Storage, StorageProvider};
    pub use smallvec::{Array, SmallVec};

    /// Represents the `smallvec::SmallVec` storage family of a particular stack capacity.
    pub struct SmallVecStorage<const CAP: usize>;

    impl<const CAP: usize, T> StorageProvider<T> for SmallVecStorage<CAP>
    where
        T: Clone,
        [T; CAP]: Array<Item = T>,
    {
        type StorageType = SmallVec<[T; CAP]>;
        #[inline]
        fn new() -> Self {
            Self
        }

        #[inline]
        fn new_storage(&mut self) -> Self::StorageType {
            Self::StorageType::new()
        }

        #[inline]
        fn storage_with_capacity(&mut self, capacity: usize) -> Self::StorageType {
            SmallVec::with_capacity(capacity)
        }
    }

    impl<const CAP: usize, T> Storage<T> for SmallVec<[T; CAP]>
    where
        T: Clone,
        [T; CAP]: Array<Item = T>,
    {
        type Provider = SmallVecStorage<CAP>;

        #[inline]
        fn clear(&mut self) {
            self.clear();
        }

        #[inline]
        fn push(&mut self, value: T) {
            self.push(value);
        }

        #[inline]
        fn pop(&mut self) -> Option<T> {
            self.pop()
        }

        #[inline]
        fn len(&self) -> usize {
            self.len()
        }

        #[inline]
        fn capacity(&self) -> usize {
            self.capacity()
        }

        #[inline]
        fn as_slice(&self) -> &[T] {
            self.as_slice()
        }

        #[inline]
        fn as_mut_slice(&mut self) -> &mut [T] {
            self.as_mut_slice()
        }
    }
}
