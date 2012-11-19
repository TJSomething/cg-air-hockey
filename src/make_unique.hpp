#ifndef MAKE_UNIQUE_H
#define MAKE_UNIQUE_H

#include <memory>
namespace std {
    // Thanks to FredOverflow
    template<typename T, typename... Args>
    unique_ptr<T> make_unique(Args&&... args)
    {
        return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
    }
}

#endif

