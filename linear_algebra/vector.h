#pragma once

#include "../common/scalar.h"
#include "contiguousparallelpartition.h"
#include <cassert>
#include <iostream>
#include <memory>


template<typename T>
class Vector
{
public:
    // constructors with 1 process
    Vector();
    Vector(const Vector& other);
    Vector(Vector&& other) noexcept;
    Vector(std::initializer_list<T> init);
    explicit Vector(int size);

    // constructor taking a partition
    explicit Vector(ContiguousParallelPartition partition);
    // constructor taking an MPI communicator and the local data size
    explicit Vector(MPI_Comm communicator, int local_size);
    // constructor taking an MPI communicator and the local data for this process
    explicit Vector(MPI_Comm communicator, std::initializer_list<T> init);

    // getter for a ContiguousParallelPartition
    const ContiguousParallelPartition& partition() const;

    Vector& operator=(const Vector& other);
    Vector& operator=(Vector&& other) noexcept;

    const T& operator[](int i) const;
    T& operator[](int i);

    int size() const;
    ~Vector() = default;

    // helper function that prints the vector
    template<typename U>
    friend std::ostream& operator<<(std::ostream& os, const Vector<U>& v);

private:
    std::unique_ptr<T[]> values_;
    int size_;
    ContiguousParallelPartition partition_;
};

template<typename T>
Vector<T>::Vector() :
    values_(nullptr), size_(0), partition_(create_partition(MPI_COMM_SELF, 0)) {}

template<typename T>
Vector<T>::Vector(const Vector<T>& other)
{
    size_ = other.size_;
    values_ = std::make_unique<T[]>(size_);
    std::copy(other.values_.get(), other.values_.get() + size_, values_.get());
    partition_ = other.partition_;
}

template<typename T>
Vector<T>::Vector(Vector<T>&& other) noexcept
{
    size_ = other.size_;
    values_ = std::make_unique<T[]>(size_);
    values_ = std::move(other.values_);
    partition_ = std::move(other.partition_);
}

template<typename T>
Vector<T>::Vector(std::initializer_list<T> init)
{
    size_ = (int)init.size();
    values_ = std::make_unique<T[]>(size_);
    std::copy(init.begin(), init.end(), values_.get());
    partition_ = create_partition(MPI_COMM_SELF, size_);
}

template<typename T>
Vector<T>::Vector(int size)
{
    assert(size >= 0);
    size_ = size;
    values_ = std::make_unique<T[]>(size_);
    partition_ = create_partition(MPI_COMM_SELF, size_);
}

// constructor taking a partition
template<typename T>
Vector<T>::Vector(ContiguousParallelPartition partition)
{
    partition_ = partition;
    size_ = partition_.local_size();
    values_ = std::make_unique<T[]>(size_);
}
// constructor taking an MPI communicator and the local data size
template<typename T>
Vector<T>::Vector(MPI_Comm communicator, int local_size)
{
    size_ = local_size;
    values_ = std::make_unique<T[]>(size_);
    partition_ = create_partition(communicator, local_size);
}
// constructor taking an MPI communicator and the local data for this process
template<typename T>
Vector<T>::Vector(MPI_Comm communicator, std::initializer_list<T> init)
{
    size_ = (int)init.size();
    values_ = std::make_unique<T[]>(size_);
    std::copy(init.begin(), init.end(), values_.get());
    partition_ = create_partition(communicator, size_);
}

// getter for a ContiguousParallelPartition
template<typename T>
const ContiguousParallelPartition& Vector<T>::partition() const
{
    return partition_;
}

template<typename T>
Vector<T>& Vector<T>::operator=(const Vector<T>& other)
{
    size_ = other.size_;
    values_ = std::make_unique<T[]>(size_);
    std::copy(other.values_.get(), other.values_.get() + size_, values_.get());
    partition_ = other.partition_;
    return *this;
}

template<typename T>
Vector<T>& Vector<T>::operator=(Vector<T>&& other) noexcept
{
    size_ = other.size_;
    values_ = std::make_unique<T[]>(size_);
    values_ = std::move(other.values_);
    partition_ = std::move(other.partition_);
    return *this;
}

template<typename T>
const T& Vector<T>::operator[](int i) const
{
    assert(i >= 0 && i < size_);
    return values_[i];
}

template<typename T>
T& Vector<T>::operator[](int i)
{
    assert(i >= 0 && i < size_);
    return values_[i];
}

template<typename T>
int Vector<T>::size() const
{
    return size_;
}

// helper function that prints the vector
template<typename T>
std::ostream& operator<<(std::ostream& os, const Vector<T>& v)
{
    os << "{";
    const int size = v.size();
    for(int i = 0; i < size - 1; ++i)
    {
        os << v[i] << ", ";
    }
    os << v[size - 1] << "}" << std::endl;
    return os;
}

// additional for getting input type
template<typename T>
constexpr MPI_Datatype mpi_get_type() noexcept
{
    MPI_Datatype mpi_type = MPI_DATATYPE_NULL;
    if constexpr(std::is_same_v<T, int>)
    {
        mpi_type = MPI_INT;
    }
    if constexpr(std::is_same_v<T, float>)
    {
        mpi_type = MPI_FLOAT;
    }
    if constexpr(std::is_same_v<T, double>)
    {
        mpi_type = MPI_DOUBLE;
    }
    assert(mpi_type != MPI_DATATYPE_NULL);
    return mpi_type;
}


local saga_status, saga = pcall(require, "lspsaga")
if not saga_status then
  return
end

saga.setup({
  -- keybinds for navigation in lspsaga window
  scroll_preview = { scroll_down = "<C-f>", scroll_up = "<C-b>" },
  -- use enter to open file with definition preview
  definition = {
    edit = "<CR>",
  },
  ui = {
    colors = {
      normal_bg = "#022746",
    },
  },
})
