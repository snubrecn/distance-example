#include <chrono>
#include <cinttypes>
#include <deque>
#include <iostream>
#include <vector>

#include "Eigen/Geometry"
#include "opencv2/opencv.hpp"

cv::Mat ConvertMapToCvMat(const int width, const int height,
                          uint8_t* const data) {
  cv::Mat image(height, width, CV_8UC1);
  image.data = data;
  return image;
}

cv::Mat ConvertDistanceMapToCvMat(const int width, const int height,
                                  const float max_distance,
                                  const float* const data) {
  cv::Mat image(height, width, CV_8UC3);
  cv::Vec3b close_color(0, 255, 0);
  cv::Vec3b distant_color(0, 0, 255);
  cv::Vec3b object_color(255, 0, 0);
  for (auto y = 0; y < height; ++y) {
    for (auto x = 0; x < width; ++x) {
      const auto ratio = data[x + width * y] / max_distance;

      const auto value =
          ratio == 0 ? object_color
                     : (1 - ratio) * close_color + ratio * distant_color;
      image.at<cv::Vec3b>(y, x) = value;
    }
  }
  return image;
}

std::vector<Eigen::Vector2i> GenerateNeighborPositions() {
  std::vector<Eigen::Vector2i> neighbor_positions;
  for (auto dy = -1; dy <= 1; ++dy) {
    for (auto dx = -1; dx <= 1; ++dx) {
      if (dx == 0 && dy == 0) continue;
      neighbor_positions.push_back(Eigen::Vector2i(dx, dy));
    }
  }
  return neighbor_positions;
}

std::vector<float> GenerateDistanceMap(const int width, const int height,
                                       const std::vector<uint8_t>& map) {
  std::vector<float> distance_map(width * height,
                                  std::numeric_limits<float>::max());
  const auto neighbor_positions = GenerateNeighborPositions();

  auto to_flat_index = [width](const Eigen::Vector2i& grid_index) {
    return grid_index.x() + width * grid_index.y();
  };
  auto to_grid_index = [width](const int flat_index) {
    return Eigen::Vector2i(flat_index % width, flat_index / width);
  };
  auto is_within_map = [width, height](const Eigen::Vector2i& grid_index) {
    return grid_index.x() >= 0 && grid_index.x() < width &&
           grid_index.y() >= 0 && grid_index.y() < height;
  };

  for (auto y = 0; y < height; ++y) {
    for (auto x = 0; x < width; ++x) {
      const Eigen::Vector2i position(x, y);
      const auto flat_index = to_flat_index(position);
      if (map[flat_index]) continue;
      distance_map[flat_index] = 0.f;

      std::deque<Eigen::Vector2i> bfs;
      bfs.push_back(position);

      while (!bfs.empty()) {
        const Eigen::Vector2i bfs_position = bfs.front();
        bfs.pop_front();
        for (const auto& neighbor_position : neighbor_positions) {
          const Eigen::Vector2i next_position =
              bfs_position + neighbor_position;
          const auto next_flat_index = to_flat_index(next_position);
          if (!is_within_map(next_position)) continue;
          if (map[next_flat_index] == 0) continue;
          const auto distance = (position - next_position).cast<float>().norm();
          if (distance < distance_map[next_flat_index]) {
            distance_map[next_flat_index] = distance;
            bfs.push_back(next_position);
          }
        }
      }
    }
  }
  return distance_map;
}

std::vector<float> ApplyDistanceTransform1D(const int width,
                                            const std::vector<float>& map) {
  constexpr auto kFloatMax = std::numeric_limits<float>::max();
  std::vector<float> distance_map(width);

  auto compute_intersection = [](const int q, const int p,
                                 const float* const data) {
    return (q * q + data[q] - p * p - data[p]) / (2 * q - 2 * p);
  };

  struct Parabola {
    int vertex_index{0};
    float start{kFloatMax};
  };
  std::vector<Parabola> envelope(width);

  envelope[0].vertex_index = 0;
  envelope[0].start = -kFloatMax;
  envelope[1].start = kFloatMax;
  int k = 0;

  for (auto q = 1; q < width; ++q) {
    // std::cerr << "q: " << q << std::endl;
    while (true) {
      float s = compute_intersection(q, envelope[k].vertex_index, map.data());
      if (s <= envelope[k].start)
        k--;
      else {
        envelope[++k].vertex_index = q;
        envelope[k].start = s;
        envelope[k + 1].start = kFloatMax;
        // std::cerr << "breaks at k: " << k << std::endl;
        break;
      }
    }
  }
  k = 0;
  for (auto q = 0; q < width; ++q) {
    while (envelope[k + 1].start < q) k++;
    distance_map[q] =
        std::hypot(q - envelope[k].vertex_index, map[envelope[k].vertex_index]);
    }
  return distance_map;
}

std::vector<float> ApplyDistanceTransform2D(const int width, const int height,
                                            const std::vector<uint8_t>& map) {
  std::vector<float> distance_map(width * height);
  constexpr auto kFloatMax = std::numeric_limits<float>::max();
  int size = std::max(width, height);
  std::vector<float> float_map(size), distance_map_1d(size);

  for (auto y = 0; y < height; ++y) {
    auto* ptr = &map[y * width];
    for (auto x = 0; x < width; ++x) float_map[x] = *ptr++ ? kFloatMax : 0.f;
    distance_map_1d = ApplyDistanceTransform1D(width, float_map);
    memcpy(&distance_map[y * width], &distance_map_1d[0],
           width * sizeof(float));
  }

  for (auto x = 0; x < width; ++x) {
    auto* ptr = &distance_map[x];
    for (auto y = 0; y < height; y++, ptr += width) float_map[y] = *ptr;
    distance_map_1d = ApplyDistanceTransform1D(height, float_map);
    ptr = &distance_map[x];
    for (auto y = 0; y < height; y++, ptr += width) *ptr = distance_map_1d[y];
  }
  return distance_map;
}

int main(void) {
  const auto width = 200;
  const auto height = 200;
  std::vector<uint8_t> map;
  map.reserve(width * height);

  for (auto y = 0; y < height; ++y)
    for (auto x = 0; x < width; ++x) map[x + width * y] = 255;

  for (auto y = 75; y < 125; ++y)
    for (auto x = 75; x < 125; ++x) map[x + width * y] = 0;

  auto ts = std::chrono::steady_clock::now();
  const auto distance_map_pg = GenerateDistanceMap(width, height, map);
  auto tm = std::chrono::steady_clock::now();
  const auto distance_map_dt = ApplyDistanceTransform2D(width, height, map);
  auto te = std::chrono::steady_clock::now();
  auto pg_duration = (tm - ts).count() * 1e-6;
  auto dt_duration = (te - tm).count() * 1e-6;
  std::cerr << "PG duration: " << pg_duration << " ms\n";
  std::cerr << "DT duration: " << dt_duration << " ms\n";

  auto max_distance = 0.f;
  for (const auto& distance : distance_map_pg)
    max_distance = std::max(max_distance, distance);

  cv::Mat map_image = ConvertMapToCvMat(width, height, map.data());
  cv::Mat distance_map_image = ConvertDistanceMapToCvMat(
      width, height, max_distance, distance_map_pg.data());
  max_distance = 0.f;
  for (const auto& distance : distance_map_dt)
    max_distance = std::max(max_distance, distance);
  cv::Mat distance_map_image_dt = ConvertDistanceMapToCvMat(
      width, height, max_distance, distance_map_dt.data());

  cv::imshow("map", map_image);
  cv::imshow("distance_map_pg", distance_map_image);
  cv::imshow("distance_map_dt", distance_map_image_dt);
  cv::imwrite("dt_pg.png", distance_map_image);
  cv::imwrite("dt_dm.png", distance_map_image_dt);
  cv::waitKey(0);
  return 0;
}