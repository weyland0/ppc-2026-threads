#include "zyazeva_s_matrix_mult_cannon_alg/omp/include/ops_omp.hpp"

#include <omp.h>

#include <cmath>
#include <cstddef>
#include <vector>

#include "zyazeva_s_matrix_mult_cannon_alg/common/include/common.hpp"

namespace zyazeva_s_matrix_mult_cannon_alg {

ZyazevaSMatrixMultCannonAlgOMP::ZyazevaSMatrixMultCannonAlgOMP(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = {};
}

bool ZyazevaSMatrixMultCannonAlgOMP::ValidationImpl() {
  const size_t sz = std::get<0>(GetInput());
  const auto &m1 = std::get<1>(GetInput());
  const auto &m2 = std::get<2>(GetInput());

  if (sz <= 0 || m1.size() != sz * sz || m2.size() != sz * sz) {
    return false;
  }

  return IsPerfectSquare(static_cast<int>(sz));
}

bool ZyazevaSMatrixMultCannonAlgOMP::PreProcessingImpl() {
  const auto sz = static_cast<int>(std::get<0>(GetInput()));
  const auto &m1 = std::get<1>(GetInput());
  const auto &m2 = std::get<2>(GetInput());

  sz_ = sz;
  matrix_a_ = m1;
  matrix_b_ = m2;
  matrix_c_.assign(static_cast<size_t>(sz_) * sz_, 0.0);

  block_sz_ = GetBlockSize(static_cast<int>(std::sqrt(sz_)));

  return true;
}

bool ZyazevaSMatrixMultCannonAlgOMP::PostProcessingImpl() {
  GetOutput() = matrix_c_;
  return true;
}

bool ZyazevaSMatrixMultCannonAlgOMP::IsPerfectSquare(int x) {
  int root = static_cast<int>(std::sqrt(x));
  return root * root == x;
}

int ZyazevaSMatrixMultCannonAlgOMP::GetBlockSize(int n) {
  for (int k = n / 2; k >= 2; --k) {
    if (n % k == 0) {
      return k;
    }
  }
  return 1;
}

void ZyazevaSMatrixMultCannonAlgOMP::CopyBlock(const std::vector<double> &matrix, std::vector<double> &block, int start,
                                               int root, int block_sz) {
  for (int i = 0; i < block_sz; ++i) {
    for (int j = 0; j < block_sz; ++j) {
      int index = start + (i * root) + j;
      block[static_cast<size_t>(i) * block_sz + j] = matrix[static_cast<size_t>(index)];
    }
  }
}

void ZyazevaSMatrixMultCannonAlgOMP::ShiftRow(std::vector<double> &matrix, int root, int row, int shift) {
  shift = shift % root;
  std::vector<double> tmp(static_cast<size_t>(root));

  for (int j = 0; j < root; ++j) {
    tmp[static_cast<size_t>(j)] = matrix[static_cast<size_t>(row) * root + ((j + shift) % root)];
  }
  for (int j = 0; j < root; ++j) {
    matrix[static_cast<size_t>(row) * root + j] = tmp[static_cast<size_t>(j)];
  }
}

void ZyazevaSMatrixMultCannonAlgOMP::ShiftColumn(std::vector<double> &matrix, int root, int col, int shift) {
  shift = shift % root;
  std::vector<double> tmp(static_cast<size_t>(root));

  for (int i = 0; i < root; ++i) {
    tmp[static_cast<size_t>(i)] = matrix[static_cast<size_t>(((i + shift) % root)) * root + col];
  }
  for (int i = 0; i < root; ++i) {
    matrix[static_cast<size_t>(i) * root + col] = tmp[static_cast<size_t>(i)];
  }
}

void ZyazevaSMatrixMultCannonAlgOMP::InitializeShift(std::vector<double> &matrix, int root, int grid_size, int block_sz,
                                                     bool is_row_shift) {
  for (int b = 0; b < grid_size; ++b) {
    for (int index = b * block_sz; index < (b + 1) * block_sz; ++index) {
      for (int shift = 0; shift < b; ++shift) {
        if (is_row_shift) {
          ShiftRow(matrix, root, index, block_sz);
        } else {
          ShiftColumn(matrix, root, index, block_sz);
        }
      }
    }
  }
}

void ZyazevaSMatrixMultCannonAlgOMP::ShiftBlocksLeft(std::vector<double> &matrix, int root, int block_sz) {
  int p = root / block_sz;

  for (int bi = 0; bi < p; ++bi) {
    std::vector<double> first_block(static_cast<size_t>(block_sz) * block_sz);

    for (int i = 0; i < block_sz; ++i) {
      for (int j = 0; j < block_sz; ++j) {
        first_block[static_cast<size_t>(i) * block_sz + j] =
            matrix[static_cast<size_t>((bi * block_sz + i)) * root + j];
      }
    }

    for (int bj = 0; bj < p - 1; ++bj) {
      for (int i = 0; i < block_sz; ++i) {
        for (int j = 0; j < block_sz; ++j) {
          matrix[static_cast<size_t>((bi * block_sz + i)) * root + (bj * block_sz) + j] =
              matrix[static_cast<size_t>((bi * block_sz + i)) * root + ((bj + 1) * block_sz) + j];
        }
      }
    }

    for (int i = 0; i < block_sz; ++i) {
      for (int j = 0; j < block_sz; ++j) {
        matrix[static_cast<size_t>((bi * block_sz + i)) * root + ((p - 1) * block_sz) + j] =
            first_block[static_cast<size_t>(i) * block_sz + j];
      }
    }
  }
}

void ZyazevaSMatrixMultCannonAlgOMP::ShiftBlocksUp(std::vector<double> &matrix, int root, int block_sz) {
  int p = root / block_sz;

  for (int bj = 0; bj < p; ++bj) {
    std::vector<double> first_block(static_cast<size_t>(block_sz) * block_sz);

    for (int i = 0; i < block_sz; ++i) {
      for (int j = 0; j < block_sz; ++j) {
        first_block[static_cast<size_t>(i) * block_sz + j] =
            matrix[static_cast<size_t>(i) * root + (bj * block_sz) + j];
      }
    }

    for (int bi = 0; bi < p - 1; ++bi) {
      for (int i = 0; i < block_sz; ++i) {
        for (int j = 0; j < block_sz; ++j) {
          matrix[static_cast<size_t>((bi * block_sz + i)) * root + (bj * block_sz) + j] =
              matrix[static_cast<size_t>(((bi + 1) * block_sz + i)) * root + (bj * block_sz) + j];
        }
      }
    }

    for (int i = 0; i < block_sz; ++i) {
      for (int j = 0; j < block_sz; ++j) {
        matrix[static_cast<size_t>(((p - 1) * block_sz + i)) * root + (bj * block_sz) + j] =
            first_block[static_cast<size_t>(i) * block_sz + j];
      }
    }
  }
}

bool ZyazevaSMatrixMultCannonAlgOMP::RunImpl() {
  int root = static_cast<int>(std::sqrt(static_cast<double>(sz_)));
  int grid_size = root / block_sz_;

  InitializeShift(matrix_a_, root, grid_size, block_sz_, true);
  InitializeShift(matrix_b_, root, grid_size, block_sz_, false);

  for (int step = 0; step < grid_size; ++step) {
#pragma omp parallel for schedule(static)
    for (int bi = 0; bi < root / block_sz_; ++bi) {
      std::vector<double> local_block_a(static_cast<size_t>(block_sz_) * block_sz_, 0.0);
      std::vector<double> local_block_b(static_cast<size_t>(block_sz_) * block_sz_, 0.0);

      for (int bj = 0; bj < root / block_sz_; ++bj) {
        int start = ((bi * block_sz_) * root) + (bj * block_sz_);

        CopyBlock(matrix_a_, local_block_a, start, root, block_sz_);
        CopyBlock(matrix_b_, local_block_b, start, root, block_sz_);

        for (int i = 0; i < block_sz_; ++i) {
          for (int k = 0; k < block_sz_; ++k) {
            double a_ik = local_block_a[static_cast<size_t>(i) * block_sz_ + k];
            for (int j = 0; j < block_sz_; ++j) {
              int index = ((bi * block_sz_ + i) * root) + (bj * block_sz_ + j);
#pragma omp atomic
              matrix_c_[static_cast<size_t>(index)] += a_ik * local_block_b[static_cast<size_t>(k) * block_sz_ + j];
            }
          }
        }
      }
    }

    ShiftBlocksLeft(matrix_a_, root, block_sz_);
    ShiftBlocksUp(matrix_b_, root, block_sz_);
  }

  return true;
}

}  // namespace zyazeva_s_matrix_mult_cannon_alg
