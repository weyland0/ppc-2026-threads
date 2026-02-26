#pragma once

#include "example_threads/common/include/common.hpp"
#include "task/include/task.hpp"

namespace chernykh_s_trapezoidal_integration {

class ChernykhSTrapezoidalIntegrationSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit ChernykhSTrapezoidalIntegrationSEQ(const InType &in);

 private:
  void RecoursiveMethod(size_t dim, std::vector<double>& current_point, double current_coeff,const InType& input, double &total_sum);
  // size_t dim - индекс оси, по какой мы идем
  // std::vector<double>& current_point - сюда записываем вычисленные координаты
  // double current_coeff - вес для текущей точки
  // const InType& Input - входные данные
  // double& total_sum - копилка сумм

  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace chernykh_s_trapezoidal_integration
