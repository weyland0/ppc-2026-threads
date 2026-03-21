#pragma once

#include <vector>

#include "task/include/task.hpp"
#include "zyazeva_s_matrix_mult_cannon_alg/common/include/common.hpp"

#ifdef _OPENMP
#  include <omp.h>
#endif

namespace zyazeva_s_matrix_mult_cannon_alg {

class ZyazevaSMatrixMultCannonAlgOMP : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kOMP;
  }
  explicit ZyazevaSMatrixMultCannonAlgOMP(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  static bool IsPerfectSquare(int x);
  static int GetBlockSize(int n);
  static void CopyBlock(const std::vector<double> &matrix, std::vector<double> &block, int start, int root,
                        int block_sz);
  static void ShiftRow(std::vector<double> &matrix, int root, int row, int shift);
  static void ShiftColumn(std::vector<double> &matrix, int root, int col, int shift);
  static void InitializeShift(std::vector<double> &matrix, int root, int grid_size, int block_sz, bool is_row_shift);
  static void ShiftBlocksLeft(std::vector<double> &matrix, int root, int block_sz);
  static void ShiftBlocksUp(std::vector<double> &matrix, int root, int block_sz);

  int sz_;
  std::vector<double> matrix_a_;
  std::vector<double> matrix_b_;
  std::vector<double> matrix_c_;
  int block_sz_;
};

}  // namespace zyazeva_s_matrix_mult_cannon_alg
