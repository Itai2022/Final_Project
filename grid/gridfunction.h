#pragma once

#include "../linear_algebra/vector.h"
#include "grid.h"
#include <functional>
#include <memory>

template<typename T>
class GridFunction
{
private:
    std::shared_ptr<RegularGrid> grid_; // because sometimes we need to construct two Gridfunctions on the same Grid
    std::function<T(const Point&)> function_;
    std::unique_ptr<Vector<T>> values_;

public:
    GridFunction() = default;
    ~GridFunction() = default;

    // konstruiere RegularGrid und dementsprechende GridFunction
    GridFunction(const RegularGrid& grid, T values);
    GridFunction(const RegularGrid& grid, const std::function<T(const Point&)>& function);
    GridFunction(const RegularGrid& grid, const Vector<T>& values);

    RegularGrid grid() const;

    T value(int node_index) const;
};
template<typename T>
GridFunction<T>::GridFunction(const RegularGrid& grid, T values)
{
    grid_ = std::make_shared<RegularGrid>(grid);
    function_ = [=](const Point&) -> T { return values; };
    values_ = nullptr;
}
template<typename T>
GridFunction<T>::GridFunction(const RegularGrid& grid, const std::function<T(const Point&)>& function)
{
    grid_ = std::make_shared<RegularGrid>(grid);
    function_ = function;
    values_ = nullptr;
}

template<typename T>
GridFunction<T>::GridFunction(const RegularGrid& grid, const Vector<T>& values)
{
    grid_ = std::make_shared<RegularGrid>(grid);
    values_ = std::make_unique<Vector<T>>(values);
}

template<typename T>
RegularGrid GridFunction<T>::grid() const
{
    return *(grid_);
}

template<typename T>
T GridFunction<T>::value(int local_node_index) const
{
    if(values_ != nullptr)
    {
        return (*(values_))[local_node_index];
    }
    else
    {
        const auto global_node_index = grid_->partition().to_global_index(local_node_index);
        return function_(grid_->node_coordinates(global_node_index));
    }
}

/*
Meine Gridfunction hat drei Mitglieder, um diskrete Funktion darzustellen. Das std::shared_ptr<RegularGrid> grid_ richtet sich auf RegularGrid.
Das std::function<T(const Point&)> function_ und das std::unique_ptr<Vector<T>> values_ behandeln zwei verschiedene Fälle.
Die function_ wird benutzt, wenn wir schon eine analytische Funktion haben und sie diskretisieren wollen.
Der Wert der Funktion auf jeden Punkt von RegularGrid wird durch value(int node_index) mit function_() berechnet.
Das values_ wird benutzt, wenn wir eine diskrete Funktion haben wie zum Beispiel die diskrete Lösung der Poisson-Gleichung.
Die diskrete Lösung wird bei uns als einen Vector<T> gespeichert, und dessen Werte passen zu den entsprechenden node_index.
Um die Werte in Vector<T> aufzurufen, return (*(values_))[node_index] in value(int node_index), falls values_ != nullptr.

Meiner Meinung nach ist diese Implementierung effizient, weil wir keine Daten mehr brauchen, um die Werte der Funktion zu speichern,
falls die analytische Funktion gegeben wird. Und der Vector<T> ist auf diesen Fall ein Nullptr ohne viel Raum zuzuteilen.
Wir haben es uns auch überlegt, dass die diskrete Lösung durch std::function dargestellt wird. Aber praktisch ist bei uns besser, die diskrete Lösung in Gridfunction als smart pointer zu speichern.

Template parameter kontroliert here return-type der Funktion und save-type von Vector<T>, nämlich von den Werte der diskreten Funktion.
Die Funktion grid() würde uns das Instance zurückgeben, sodass die Funktional von RegularGrid beim Schreiben von vtk und poisson-matrix benutzt werden können.
*/