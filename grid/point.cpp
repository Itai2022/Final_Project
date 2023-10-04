
#include "../common/equals.h"

#include "point.h"

#include <cassert>

Point::Point(scalar_t value)
{
    for(int i = 0; i < space_dimension; i++)
    {
        coordinates_[i] = value;
    }
}

Point::Point(scalar_t x, scalar_t y)
{
    assert(space_dimension == 2);
    coordinates_[0] = x;
    coordinates_[1] = y;
}

Point::Point(scalar_t x, scalar_t y, scalar_t z)
{
    assert(space_dimension == 3);
    coordinates_[0] = x;
    coordinates_[1] = y;
    coordinates_[2] = z;
}

int Point::size() const
{
    return space_dimension;
}

const scalar_t& Point::operator[](int i) const
{
    assert(i >= 0 && i < space_dimension);
    return coordinates_[i];
}

scalar_t& Point::operator[](int i)
{
    assert(i >= 0 && i < space_dimension);
    return coordinates_[i];
}

bool equals(const Point& lhs, const Point& rhs)
{
    for(int i = 0; i < space_dimension; ++i)
    {
        if(!equals(lhs[i], rhs[i])) return false;
    }
    return true;
}