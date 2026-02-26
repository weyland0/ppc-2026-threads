#include "example_threads/seq/include/ops_seq.hpp"

#include <numeric>
#include <vector>

#include "example_threads/common/include/common.hpp"
#include "util/include/util.hpp"

namespace chernykh_s_trapezoidal_integration {

ChernykhSTrapezoidalIntegrationSEQ::ChernykhSTrapezoidalIntegrationSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

void ChernykhSTrapezoidalIntegrationSEQ::RecoursiveMethod(size_t dim, std::vector<double>& current_point, double current_coeff,const InType& input, double &total_sum) {
  // size_t dims - индекс оси, по какой мы идем
  // std::vector<double>& current_point - сюда записываем вычисленные координаты
  // double current_coeff - вес для текущей точки
  // const InType& Input - входные данные
  // double& total_sum - копилка сумм
  if(dim == input.limits.size()){
    total_sum+=input.func(current_point)*current_coeff;
    return;
  }
  double a = input.limits[dim].first; 
  double b = input.limits[dim].second; // координаты на текущей оси
  size_t n = input.steps[dim]; // количество разбиений на этой оси
  double h = (b-a)/n; // длина текущего шага

  for (int i = 0; i<=n;i++){
    current_point[dim] = a + i*h; 
    double local_coeff = (i==0 || i==n) ? 0.5 : 1.0;
    RecoursiveMethod(dim+1, current_point, local_coeff*current_coeff, input, total_sum);
  }


}



bool ChernykhSTrapezoidalIntegrationSEQ::ValidationImpl() {
  const auto &input = GetInput();

  if(input.limits.empty()){
    return false;
  }

  if(input.limits.size() != input.steps.size()){
    return false;
  }
  for(auto s : input.steps){
    if (s<=0){
      return false;
    }
  }

  return true;
}

bool ChernykhSTrapezoidalIntegrationSEQ::PreProcessingImpl() {
  
  return true;
}

bool ChernykhSTrapezoidalIntegrationSEQ::RunImpl() {
  const auto& input = GetInput();
  size_t dims = input.limits.size();
  std::vector<double> current_point(dims); 
  double total_sum = 0.0;
  RecursiveMethod(0, current_point, 1.0, input, total_sum);
  double h_prod = 1.0;
  for (size_t i = 0; i < dims; ++i) {
    h_prod *= (input.limits[i].second - input.limits[i].first) / input.steps[i];
  }
  GetOutput() = total_sum * h_prod;

  return true;
}

bool ChernykhSTrapezoidalIntegrationSEQ::PostProcessingImpl() {
  
  return true;
}

}  // namespace chernykh_s_trapezoidal_integration
