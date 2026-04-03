#pragma once

#include <algorithm>
#include <cmath>
#include <type_traits>

namespace nep_gpu_kokkos_utils {

inline void compute_box_thickness(const double h[9], double thickness[3])
{
  // h is row-major of column vectors: [ax bx cx ay by cy az bz cz]
  const double a[3] = {h[0], h[3], h[6]};
  const double b[3] = {h[1], h[4], h[7]};
  const double c[3] = {h[2], h[5], h[8]};

  const auto dot3 = [](const double u[3], const double v[3]) { return u[0] * v[0] + u[1] * v[1] + u[2] * v[2]; };
  const auto cross3 = [](const double u[3], const double v[3], double w[3]) {
    w[0] = u[1] * v[2] - u[2] * v[1];
    w[1] = u[2] * v[0] - u[0] * v[2];
    w[2] = u[0] * v[1] - u[1] * v[0];
  };

  double bxc[3], cxa[3], axb[3];
  cross3(b, c, bxc);
  cross3(c, a, cxa);
  cross3(a, b, axb);
  const double volume = std::abs(dot3(a, bxc));
  thickness[0] = volume / std::sqrt(dot3(bxc, bxc));
  thickness[1] = volume / std::sqrt(dot3(cxa, cxa));
  thickness[2] = volume / std::sqrt(dot3(axb, axb));
}

inline void invert3x3_rowmajor(const double h[9], double hinv[9])
{
  const double det =
    h[0] * (h[4] * h[8] - h[5] * h[7]) -
    h[1] * (h[3] * h[8] - h[5] * h[6]) +
    h[2] * (h[3] * h[7] - h[4] * h[6]);
  const double invdet = 1.0 / det;

  hinv[0] =  (h[4] * h[8] - h[5] * h[7]) * invdet;
  hinv[1] = -(h[1] * h[8] - h[2] * h[7]) * invdet;
  hinv[2] =  (h[1] * h[5] - h[2] * h[4]) * invdet;
  hinv[3] = -(h[3] * h[8] - h[5] * h[6]) * invdet;
  hinv[4] =  (h[0] * h[8] - h[2] * h[6]) * invdet;
  hinv[5] = -(h[0] * h[5] - h[2] * h[3]) * invdet;
  hinv[6] =  (h[3] * h[7] - h[4] * h[6]) * invdet;
  hinv[7] = -(h[0] * h[7] - h[1] * h[6]) * invdet;
  hinv[8] =  (h[0] * h[4] - h[1] * h[3]) * invdet;
}

template<class ViewType>
struct NepKokkosAosTraits {
  using value_type = typename ViewType::non_const_value_type;
  using layout_type = typename ViewType::array_layout;
  static constexpr bool is_layout_right = std::is_same<layout_type, Kokkos::LayoutRight>::value;
  static constexpr bool is_double = std::is_same<value_type, double>::value;
  static constexpr bool direct_ok = is_layout_right && is_double;
};

template<bool direct_ok, class DeviceType, class ViewType>
struct NepXyzPtr;

template<class DeviceType, class ViewType>
struct NepXyzPtr<true, DeviceType, ViewType> {
  static const double* get(
    const typename DeviceType::execution_space& /*exec*/,
    const ViewType& d_x,
    Kokkos::View<double*, DeviceType>& /*d_xyz_aos*/,
    int /*nall*/,
    bool& /*did_pack*/,
    const char* /*label*/)
  {
    return d_x.data();
  }
};

template<class DeviceType, class ViewType>
struct NepXyzPtr<false, DeviceType, ViewType> {
  static const double* get(
    const typename DeviceType::execution_space& exec,
    const ViewType& d_x,
    Kokkos::View<double*, DeviceType>& d_xyz_aos,
    int nall,
    bool& did_pack,
    const char* label)
  {
    if (d_xyz_aos.extent_int(0) != 3 * nall) {
      d_xyz_aos = Kokkos::View<double*, DeviceType>(Kokkos::NoInit(label), 3 * nall);
    }
    auto d_xyz_aos_l = d_xyz_aos;
    Kokkos::parallel_for(
      Kokkos::RangePolicy<typename DeviceType::execution_space>(exec, 0, nall),
      KOKKOS_LAMBDA(const int i) {
        d_xyz_aos_l(3 * i + 0) = d_x(i, 0);
        d_xyz_aos_l(3 * i + 1) = d_x(i, 1);
        d_xyz_aos_l(3 * i + 2) = d_x(i, 2);
      });
    did_pack = true;
    return d_xyz_aos.data();
  }
};

} // namespace nep_gpu_kokkos_utils

