#ifndef PMSC_SPACE_DIMENSION_H
#define PMSC_SPACE_DIMENSION_H

#ifndef PMSC_SPACE_DIMENSION
constexpr inline int space_dimension = 2; // default space dimension if not set by macro
#else
constexpr inline int space_dimension = PMSC_SPACE_DIMENSION;
#endif // PMSC_SPACE_DIMENSION

#endif // PMSC_SPACE_DIMENSION_H
