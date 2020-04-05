#include "Bitmap.h"
#include "main.h"
using namespace geometry;

struct Light {
    Light(const Vec3f &p, const float i) : position(p), intensity(i) {}
    Vec3f position;
    float intensity;
};

struct Material {
    Material(const float r, const Vec4f &a, const Vec3f &color, const float spec) : refr_index(r), albedo(a), diff_color(color), spec_exp(spec) {}
    Material() : refr_index(1), albedo(1,0,0,0), diff_color(), spec_exp() {}
    float refr_index;
    Vec4f albedo;
    Vec3f diff_color;
    float spec_exp;
};

class Object{
public:
  virtual bool intersect(const Vec3f &orig, const Vec3f &dir, float &t0) = 0;
  virtual const Material getMaterial() = 0;
  virtual const Vec3f N(const Vec3f& hit) = 0;
  virtual const std::string getType() = 0;
  virtual ~Object(){}
};

class Par : public Object{
private:
  Material material;
  float l,w,h;
  Vec3f llc;
  Vec3f Min, Max;
  std::vector<Vec4f> eqs;
public:
  Par(const float length, const float width, const float height, const Vec3f left_low_corner, const Material mat) : l(length), w(width), h(height), llc(left_low_corner), material(mat)
  {
    Vec3f vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8;
    vec1 = llc;
    vec2 = Vec3f( llc.x,llc.y + h, llc.z);
    vec3 = Vec3f(llc.x,llc.y,llc.z + l);
    vec4 = Vec3f(llc.x+w,llc.y + h,llc.z);
    vec5 = Vec3f(llc.x, llc.y + h, llc.z + l);
    vec6 = Vec3f(llc.x + w,llc.y, llc.z + l);
    vec7 = Vec3f(llc.x + w, llc.y, llc.z);
    vec8 = Vec3f(llc.x + w, llc.y + h, llc.z + l);
    eqs.push_back(plain(vec1,vec2, vec4));
    eqs.push_back(plain(vec1,vec3 ,vec6));
    eqs.push_back(plain(vec2,vec3, vec5));
    eqs.push_back(plain(vec5, vec8, vec3));
    eqs.push_back(plain(vec8, vec7, vec4));
    eqs.push_back(plain(vec5, vec8, vec2));
    Min.x = std::min(llc.x, llc.x + w);
    Min.y = std::min(llc.y, llc.y + h);
    Min.z = std::min(llc.z, llc.z + l);
    Max.x = std::max(llc.x, llc.x + w);
    Max.y = std::max(llc.y, llc.y + h);
    Max.z = std::max(llc.z, llc.z + l);
  }

  const std::string getType() override {
      return "Par";
  }

  static const Vec4f plain(const Vec3f& v1, const Vec3f& v2, const Vec3f& v3){
    float x = (v2.y - v1.y)*(v3.z - v1.z) - (v2.z - v1.z)*(v3.y - v1.y);
    float y = (v2.z - v1.z)*(v3.x - v1.x) - (v2.x - v1.x)*(v3.z - v1.z);
    float z = (v2.x - v1.x)*(v3.y - v1.y) - (v2.y - v1.y)*(v3.x - v1.x);
    float wt = -v1.x*(v2.y-v1.y)*(v3.z-v1.z) + v1.z*(v2.y - v1.y)*(v3.x - v1.x) - v1.y*(v2.z - v1.z)*(v3.x - v1.x) - v1.z*(v2.x- v1.x)*(v3.y - v1.y) + v1.y*(v2.x - v1.x)*(v3.z - v1.z) + v1.x*(v2.z - v1.z)*(v3.y - v1.y);
    return Vec4f(x,y,z,wt);
  }
  bool intersect(const Vec3f &orig, const Vec3f &dir, float &t0) override{
      float tmin = INT16_MIN + 0.0;
      float tmax = std::numeric_limits<float>::max();

      if (dir.x != 0.0) {
          float tx1 = (Min.x - orig.x)/dir.x;
          float tx2 = (Max.x - orig.x)/dir.x;

          tmin = std::max(tmin, std::min(tx1, tx2));
          tmax = std::min(tmax, std::max(tx1, tx2));
      }
      if (tmax < tmin)
          return false;

      if (dir.y != 0.0) {
          float ty1 = (Min.y - orig.y)/dir.y;
          float ty2 = (Max.y - orig.y)/dir.y;

          tmin = std::max(tmin, std::min(ty1, ty2));
          tmax = std::min(tmax, std::max(ty1, ty2));
      }
      if (tmax < tmin)
          return false;

      if (dir.z != 0.0) {
          float tz1 = (Min.z - orig.z)/dir.z;
          float tz2 = (Max.z - orig.z)/dir.z;

          tmin = std::max(tmin, std::min(tz1, tz2));
          tmax = std::min(tmax, std::max(tz1, tz2));
      }
      if (tmax < tmin || tmin < 0)
          return false;
      t0 = tmin;
      return true;
  }
  const Material getMaterial() override {
    return material;
  }

  const Vec3f N(const Vec3f& hit) override {
      int i;
      float x,y,z;
      for(i = 0; i < eqs.size(); i++)
      {
          if (fabs(hit.x * eqs[i].x + hit.y * eqs[i].y + hit.z * eqs[i].z + eqs[i].w) <=0.0001 ){
              z = eqs[i].z;
              y = eqs[i].y;
              x = eqs[i].x;
              if(fabs(hit.x - Max.x) <= 0.0001)
              {
                  x = -x;
              }
              if(fabs(hit.z - Max.z) <= 0.0001){
                  z = -z;
              }
              return Vec3f(x, y, z).normalize();
          }
      }
      throw "Norm err";
  }
};
class Sphere : public Object {
private:
    Vec3f center;
    float radius;
    Material material;
public:
  Sphere(const Vec3f &c, const float r, const Material &m) : center(c), radius(r), material(m) {}
  bool intersect(const Vec3f &orig, const Vec3f &dir, float &t0) override{
      Vec3f L = center - orig;
      float tca = L*dir;
      float d2 = L*L - tca*tca;
      if (d2 > radius*radius) return false;
      float thc = sqrtf(radius*radius - d2);
      t0       = tca - thc;
      float t1 = tca + thc;
      if (t0 < 0) t0 = t1;
      return t0 >= 0;
  }
  const Material getMaterial() override {
    return material;
  }
  const Vec3f N(const Vec3f& hit) override {
    return (hit - center).normalize();
  }
  const std::string getType() override {
      return "Sphere";
  }
};

//static params
float Rout = 1000;
Material envm;
Sphere senv(Vec3f(0,0,0), Rout, envm);

class Scene;

Vec3f cast_ray(const Vec3f &orig, const Vec3f &dir,const Scene& scene, const size_t& _i, const size_t& j, size_t depth=0);

class Scene{
private:
    std::vector<Light*> lights;
    std::vector<Object*> objects;
public:
    Scene(){};
    void addLight(Light* light){
        lights.push_back(light);
    }
    void addObject(Object* object){
        objects.push_back(object);
    }
    bool intersect(const Vec3f &orig, const Vec3f &dir,  Vec3f &hit, Vec3f &N, Material &material, std::string& type, int& id) const {
        float obj_t = std::numeric_limits<float>::max();
        for (size_t i=0; i < objects.size(); i++) {
            float t_i;
            if (objects[i]->intersect(orig, dir, t_i) && t_i < obj_t) {
                obj_t = t_i;
                type = objects[i]->getType();
                hit = orig + dir*t_i;
                id = i;
                N = objects[i]->N(hit);
                material = objects[i]->getMaterial();
            }
        }

        float check_t = std::numeric_limits<float>::max();
        if (fabs(dir.y)>1e-3)  {
            float t = -(orig.y + 8)/dir.y;
            Vec3f pt = orig + dir*t;
            if (t>0 && t<obj_t) {
                type = "Checkerboard";
                check_t = t;
                hit = pt;
                id = -1;
                N = Vec3f(0,1,0);
                material.diff_color = (int(0.25*hit.x + Rout) + int(0.25*hit.z)) & 1 ? Vec3f(.0, .0, .0) : Vec3f(.6, .6, .6);
            }
        }
        return std::min(obj_t, check_t)<1000;
    }

    const std::vector<Light *> & getLights() const {
        return lights;
    }

    static void smooth(std::vector<Vec3f>& buf, const int width = 512, const int height = 512, const float eps = 0.6){
        std::vector<Vec3f> alter(width * height);
#pragma omp parallel for
        for(size_t i = 1; i < width-1; i++){
            for (size_t j = 1; j < height-1; j++) {
                float x = (buf[i - 1 + j * width].x + buf[i + 1 + j * width].x + buf[i + (j - 1) * width].x +
                           buf[i - 1 + (j + 1) * width].x + buf[i - 1 + (j-1) * width].x + buf[i - 1 + (j + 1) * width].x + buf[i+1 + (j - 1)* width].x + buf[i + 1 + (j +1 ) * width].x) / 8;
                float y = (buf[i - 1 + j * width].y + buf[i + 1 + j * width].y + buf[i + (j - 1) * width].y +
                           buf[i - 1 + (j + 1) * width].y + buf[i - 1 + (j-1) * width].y + buf[i - 1 + (j + 1) * width].y + buf[i+1 + (j - 1)* width].y + buf[i + 1 + (j +1 ) * width].y) / 8;
                float z = (buf[i - 1 + j * width].z + buf[i + 1 + j * width].z + buf[i + (j - 1) * width].z +
                           buf[i - 1 + (j + 1) * width].z + buf[i - 1 + (j-1) * width].z + buf[i - 1 + (j + 1) * width].z + buf[i+1 + (j - 1)* width].z + buf[i + 1 + (j +1 ) * width].z) / 8;
                alter[i + j * width] = Vec3f(x, y, z) - buf[i + j * width];
            }
        }
#pragma omp parallel for
        for(size_t i = 1; i < width-1; i++){
            for (size_t j = 1; j < height-1; j++) {
                buf[i + j * width] = alter[i + j * width] * eps + buf[i + j * width] ;
            }
        }
    }
    void render(const int width = 512, const int height = 512, const float fov = M_PI/2., const std::string outFilePath = "out.bmp", const float smooth_eps = 0.0) {
        std::vector<Vec3f> buf(width*height);
#pragma omp parallel for
        for (size_t j = 0; j<height; j++) {
            for (size_t i = 0; i<width; i++) {
                float dir_x =  (i + 0.5) -  width/2.;
                float dir_y =  (j + 0.5) - height/2.;
                float dir_z = -height/(2.*tan(fov/2.));
                buf[i+j*width] = cast_ray(Vec3f(0,0,0), Vec3f(dir_x, dir_y, dir_z).normalize(), *this, i, j);
            }
        }
        if (smooth_eps != 0.0) {
            smooth(buf, width, height, smooth_eps);
        }
        SaveBMP(outFilePath.c_str(), buf, width, height);
    }

    ~Scene(){
        for(size_t i = 0; i < lights.size(); i++){
            delete(lights[i]);
        }
        for(size_t i = 0; i < objects.size(); i++){
            delete(objects[i]);
        }
    }
};

Vec3f outer(const Vec3f& orig, const Vec3f& dir){
    float t = 0;
    senv.intersect(orig, dir, t);
    Vec3f hitpt = orig + dir * t;
    bool star = rand()%1000 < 30;
    if (star)
        return Vec3f(1,1,1);
    return Vec3f(hitpt.y/Rout,hitpt.y/Rout,1);
}

Vec3f cast_ray(const Vec3f &orig, const Vec3f &dir,const Scene& scene, const size_t& _i, const size_t& _j, size_t depth) {
    Vec3f hitpt, N;
    Material material;
    std::string type;
    int id = -1;
    if (!scene.intersect(orig, dir, hitpt, N, material, type, id) || depth>5) {
        return outer(orig, dir);
    }
    Vec3f refl_dir = reflect(dir, N).normalize();
    Vec3f refr_dir = refract(dir, N, material.refr_index).normalize();
    Vec3f refl_orig = refl_dir*N < 0 ? hitpt - N*1e-3 : hitpt + N*1e-3;
    Vec3f refr_orig = refr_dir*N < 0 ? hitpt - N*1e-3 : hitpt + N*1e-3;
    Vec3f refl_color = cast_ray(refl_orig, refl_dir, scene, _i, _j, depth + 1);
    Vec3f refr_color = cast_ray(refr_orig, refr_dir, scene, _i, _j, depth + 1);

    float diffuse_light_intensity = 0, specular_light_intensity = 0;
    for (size_t i=0; i<scene.getLights().size(); i++) {
        Vec3f light_dir      = (scene.getLights()[i]->position - hitpt).normalize();
        float light_distance = (scene.getLights()[i]->position - hitpt).norm();
        bool lies = light_dir*N < 0;
        Vec3f shadow_orig = lies ? hitpt - N*1e-3 : hitpt + N*1e-3;
        Vec3f shadow_pt, shadow_N;
        Material tmpmaterial;
        std::string tmptype = "None";
        int tmpid = -1;
        if (scene.intersect(shadow_orig, light_dir, shadow_pt, shadow_N, tmpmaterial, tmptype, tmpid)) {
            float norm = (shadow_pt - shadow_orig).norm();
            if (norm < light_distance && tmpid!=id)
                continue;
        }
        diffuse_light_intensity  += scene.getLights()[i]->intensity * std::max(0.f, light_dir*N);
        specular_light_intensity += powf(std::max(0.f, reflect(light_dir, N)*dir), material.spec_exp)*scene.getLights()[i]->intensity;
    }
    return material.diff_color * diffuse_light_intensity * material.albedo[0] + Vec3f(1., 1., 1.)*specular_light_intensity * material.albedo[1] + refl_color*material.albedo[2] + refr_color*material.albedo[3];
}






int main(int argc, const char** argv)
{
    int nthreads = 1, width = 512, height = 512, num_objects = 4, num_lights = 1;
    float fov = M_PI/2., smooth = 0.0;
   std::unordered_map<std::string, std::string> cmdLineParams;

   for(int i=0; i<argc; i++)
   {
     std::string key(argv[i]);

     if(key.size() > 0 && key[0]=='-')
     {
       if(i != argc-1) // not last argument
       {
         cmdLineParams[key] = argv[i+1];
         i++;
       }
       else
         cmdLineParams[key] = "";
     }
   }

   std::string outFilePath = "out.bmp";
   if(cmdLineParams.find("-out") != cmdLineParams.end())
     outFilePath = cmdLineParams["-out"];

   int sceneId = 1;
   if(cmdLineParams.find("-scene") != cmdLineParams.end())
     sceneId = atoi(cmdLineParams["-scene"].c_str());
   if (cmdLineParams.find("-threads") != cmdLineParams.end()){
       nthreads = atoi(cmdLineParams["-threads"].c_str());
   }
   if (cmdLineParams.find("-width") != cmdLineParams.end()){
       width = atoi(cmdLineParams["-width"].c_str());
   }
    if (cmdLineParams.find("-height") != cmdLineParams.end()){
        width = atoi(cmdLineParams["-height"].c_str());
    }

    if (cmdLineParams.find("-smooth") != cmdLineParams.end()){
        smooth = atof(cmdLineParams["-smooth"].c_str());
    }

    if (cmdLineParams.find("-num_objects") != cmdLineParams.end()){
        num_objects = atoi(cmdLineParams["-num_objects"].c_str());
    }

    if (cmdLineParams.find("-num_lights") != cmdLineParams.end()){
        num_lights = atoi(cmdLineParams["-num_lights"].c_str());
    }

   omp_set_num_threads(nthreads);
   if(sceneId != 1)
     return 0;

    Material      ivory(1.0, Vec4f(0.6,  0.3, 0.1, 0.0), Vec3f(0.4, 0.4, 0.3),   50.);
    Material      glass(1.5, Vec4f(0.0,  0.5, 0.1, 0.8), Vec3f(0.6, 0.7, 0.8),  125.);
    Material red_rubber(1.0, Vec4f(0.9,  0.1, 0.0, 0.0), Vec3f(0.3, 0.1, 0.1),   10.);
    Material     mirror(1.0, Vec4f(0.0, 10.0, 0.8, 0.0), Vec3f(1.0, 1.0, 1.0), 1425.);
    std::vector<Object *> objects;
    objects.push_back(new Sphere(Vec3f(4,    -1,   -10), 1.5,      ivory));
    objects.push_back(new Par(5,5,5,Vec3f(-9.5,-7,-14), red_rubber));
    objects.push_back(new Sphere(Vec3f(-1.0, -1.5, -12), 2,      glass));
    objects.push_back(new Par(4,4,1,Vec3f(-2, -5, -6.5), mirror));
    objects.push_back(new Sphere(Vec3f( -2.5, 2, -21), 3, red_rubber));
    objects.push_back(new Sphere(Vec3f( 5,    5,   -18), 4,     mirror));
    objects.push_back(new Par(3,4,5,Vec3f(9,-2,-14), glass));
    std::vector<Light *> lights;
    lights.push_back(new Light(Vec3f(-20, 20,  20), 1.5));
    lights.push_back(new Light(Vec3f( 30, 50, -25), 1.8));
    lights.push_back(new Light(Vec3f( 30, 20,  30), 1.7));
    Scene scene;
    if (num_objects > 7){
        std::cout<<"Num_objects up to 7 incl\n";
        return -1;
    }
    if (num_lights > 3){
        std::cout<<"Num_lights up to 3 incl\n";
    }
    for(int i = 0; i < num_objects; i++){
        scene.addObject(objects[i]);
    }
    for(int i = 0; i < num_lights; i++){
        scene.addLight(lights[i]);
    }
    scene.render(width, height, fov, outFilePath, smooth);
    return 0;
}
