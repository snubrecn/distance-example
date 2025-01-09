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

int main(void) {
  const auto width = 200;
  const auto height = 200;
  std::vector<uint8_t> map;
  map.reserve(width * height);

  for (auto y = 0; y < height; ++y)
    for (auto x = 0; x < width; ++x) map[x + width * y] = 255;

  for (auto y = 75; y < 125; ++y)
    for (auto x = 75; x < 125; ++x) map[x + width * y] = 0;

  const auto distance_map = GenerateDistanceMap(width, height, map);

  auto max_distance = 0.f;
  for (const auto& distance : distance_map)
    max_distance = std::max(max_distance, distance);

  cv::Mat map_image = ConvertMapToCvMat(width, height, map.data());
  cv::Mat distance_map_image = ConvertDistanceMapToCvMat(
      width, height, max_distance, distance_map.data());

  cv::imshow("map", map_image);
  cv::imshow("distance_map", distance_map_image);
  cv::waitKey(0);
  return 0;
}