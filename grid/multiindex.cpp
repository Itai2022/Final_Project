#include "multiindex.h"

MultiIndex::MultiIndex(int value)
{
    for(int i = 0; i < space_dimension; i++)
    {
        values_[i] = value;
    }
}

MultiIndex::MultiIndex(int i, int j)
{
    assert(space_dimension == 2);
    values_[0] = i;
    values_[1] = j;
}

MultiIndex::MultiIndex(int i, int j, int k)
{
    assert(space_dimension == 3);
    values_[0] = i;
    values_[1] = j;
    values_[2] = k;
}

int MultiIndex::size() const
{
    return space_dimension;
}

bool MultiIndex::operator==(const MultiIndex& other) const
{
    for(int i = 0; i < space_dimension; i++)
    {
        if(values_[i] != other.values_[i])
        {
            return false;
        }
    }
    return true;
}

bool MultiIndex::operator!=(const MultiIndex& other) const
{
    return !(values_ == other.values_);
}

const int& MultiIndex::operator[](int i) const
{
    assert(i >= 0 && i < space_dimension);
    return values_[i];
}

int& MultiIndex::operator[](int i)
{
    assert(i >= 0 && i < space_dimension);
    return values_[i];
}