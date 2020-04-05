#ifndef RT_MAIN_H
#define RT_MAIN_H
#include <vector>
#include <fstream>
#include <cstring>
#include "omp.h"
#include "geometry.h"
#include <cstdint>
#include <string>
#include <unordered_map>
#include <cmath>
#include <limits>
#include <algorithm>
#include <omp.h>


struct Light;

struct Material;

class Object;

class Par;

class Sphere;

class Scene;

geometry::Vec3f cast_ray(const geometry::Vec3f &orig, const geometry::Vec3f &dir,const Scene& scene, const size_t& _i, const size_t& _j, size_t depth);

//TODO: class ray
#endif //RT_MAIN_H
