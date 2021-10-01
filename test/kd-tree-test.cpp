#include <algorithm>
#include <chrono>
#include <iostream>
#include <random>

#include <gtest/gtest.h>

#include "kd-tree.h"

class Number: public KDTreePlaceable<1> {
public:
    Number(float num) : val_(num) {
    };

    std::array<float, 1> GetPoint() const override {
        return {val_};
    }

    float GetDistanceTo(const KDTreePlaceable<1>& object) const override {
        return std::abs(val_ - object.GetPoint()[0]);
    }

    float Val() const {
        return val_;
    }

private:
    float val_;
};

std::unique_ptr<KDTree<Number>> BuildTreeNumber(int step) {
    std::vector<Number> numbers;

    for (int val = 0; val <= 10; val += step) {
        numbers.emplace_back(val);
    }

    KDTreeBuilder<Number> builder(SplitType::Median, 10);
    builder.LoadObjects(std::move(numbers));

    return builder.BuildTree();
}

TEST(TestNumber, Simple) {
    auto tree = BuildTreeNumber(2);

    auto res = tree->SearchClosest(Number(0.9), 3, 2);
    ASSERT_EQ(res.size(), 2);
    ASSERT_EQ(res[0]->Val(), 0);
    ASSERT_EQ(res[1]->Val(), 2);

    res = tree->SearchClosest(Number(5.5), 0.5, 2);
    ASSERT_EQ(res.size(), 1);
    ASSERT_EQ(res[0]->Val(), 6);

    res = tree->SearchClosest(Number(4.9), 5, 2);
    ASSERT_EQ(res.size(), 2);
    ASSERT_EQ(res[0]->Val(), 4);
    ASSERT_EQ(res[1]->Val(), 6);
}

TEST(TestNumber, Empty) {
    auto tree = BuildTreeNumber(5);

    auto res = tree->SearchClosest(Number(3), 1, 3);
    ASSERT_EQ(res.size(), 0);
}

TEST(TestNumber, Order) {
    auto tree = BuildTreeNumber(1);

    auto res = tree->SearchClosest(Number(2.9), 10, 10);
    ASSERT_EQ(res.size(), 10);
    ASSERT_EQ(res[0]->Val(), 3);
    ASSERT_EQ(res[1]->Val(), 2);
    ASSERT_EQ(res[2]->Val(), 4);
    ASSERT_EQ(res[3]->Val(), 1);

    res = tree->SearchClosest(Number(7.1), 10, 10);
    ASSERT_EQ(res.size(), 10);
    ASSERT_EQ(res[0]->Val(), 7);
    ASSERT_EQ(res[1]->Val(), 8);
    ASSERT_EQ(res[2]->Val(), 6);
    ASSERT_EQ(res[3]->Val(), 9);
}

class Point: public KDTreePlaceable<2> {
public:
    Point(float x, float y) : coord_{x, y} {
    };

    std::array<float, 2> GetPoint() const override {
        return coord_;
    }

    float GetDistanceTo(const KDTreePlaceable<2>& object) const override {
        float dist = 0;
        for (int ind = 0; ind < 2; ++ind) {
            dist += std::pow(coord_[ind] - object.GetPoint()[ind], 2);
        }
        return std::sqrt(dist);
    }

private:
    std::array<float, 2> coord_;
};

std::unique_ptr<KDTree<Point>> BuildTreePoint(int seed, int amount) {
    std::mt19937 random_generator(seed);
    std::uniform_real_distribution<float> dist(-10, 10);

    std::vector<Point> points;
    for (int val = 0; val < amount; ++val) {
        points.emplace_back(dist(random_generator), dist(random_generator));
    }

    KDTreeBuilder<Point> builder(SplitType::Median, 10);
    builder.LoadObjects(std::move(points));

    return builder.BuildTree();
}

inline void AssertPoint(const Point& point, std::array<float, 2> expected, float esp) {
    EXPECT_NEAR(point.GetPoint()[0], expected[0], esp);
    EXPECT_NEAR(point.GetPoint()[1], expected[1], esp);
}

TEST(TestPoint, Simple) {
    auto tree = BuildTreePoint(343, 10);

    auto res = tree->SearchClosest(Point(-7, -6), 1, 1);
    ASSERT_EQ(res.size(), 1);
    AssertPoint(*res[0], {-7, -6}, 0.5);
}

TEST(TestPoint, Empty) {
    auto tree = BuildTreePoint(343, 10);

    auto res = tree->SearchClosest(Point(0, 0), 1, 10);
    ASSERT_EQ(res.size(), 0);
}

TEST(TestPoint, Order) {
    auto tree = BuildTreePoint(343, 10);

    auto res = tree->SearchClosest(Point(0, 0), 10, 5);
    ASSERT_EQ(res.size(), 5);
    AssertPoint(*res[0], {1.78, 3.13}, 0.01);
    AssertPoint(*res[1], {1.46, -5.22}, 0.01);
    AssertPoint(*res[2], {-5.37, 2.92}, 0.01);
    AssertPoint(*res[3], {7.29, -1.6}, 0.01);
    AssertPoint(*res[4], {-4.5, 8.42}, 0.01);

    res = tree->SearchClosest(Point(0, 0), 10, 10);
    ASSERT_EQ(res.size(), 10);
    AssertPoint(*res[0], {1.78, 3.13}, 0.01);
    AssertPoint(*res[1], {1.46, -5.22}, 0.01);
    AssertPoint(*res[2], {-5.37, 2.92}, 0.01);
    AssertPoint(*res[3], {7.29, -1.6}, 0.01);
    AssertPoint(*res[4], {-4.5, 8.42}, 0.01);

    auto target = Point(0, 0);
    auto comparator = [&target](const Point& first, const Point& second) {
        return first.GetDistanceTo(target) > second.GetDistanceTo(target);
    };
    res = tree->SearchClosest(target, 10, 10, comparator);
    ASSERT_EQ(res.size(), 10);
    AssertPoint(*res[9], {1.78, 3.13}, 0.01);
    AssertPoint(*res[8], {1.46, -5.22}, 0.01);
    AssertPoint(*res[7], {-5.37, 2.92}, 0.01);
    AssertPoint(*res[6], {7.29, -1.6}, 0.01);
    AssertPoint(*res[5], {-4.5, 8.42}, 0.01);
}

class Quad: public KDTreePlaceable<4> {
public:
    Quad(float first, float second, float third, float forth): val_({first, second, third, forth}) {
    }

    std::array<float, 4> GetPoint() const override {
        return val_;
    }

    float GetDistanceTo(const KDTreePlaceable<4>& object) const override {
        float dist = 0;
        for (int ind = 0; ind < 4; ++ind) {
            dist += std::pow(val_[ind] - object.GetPoint()[ind], 2);
        }
        return std::sqrt(dist);
    }

private:
    std::array<float, 4> val_;
};

std::vector<Quad> CreateQuads(std::mt19937& gen, int amount) {
    std::vector<Quad> quads;
    std::uniform_real_distribution<float> val_dist(-1000, 1000);

    quads.reserve(amount);
    for (int ind = 0; ind < amount; ++ind) {
        quads.emplace_back(val_dist(gen), val_dist(gen), val_dist(gen), val_dist(gen));
    }
    return quads;
}

Quad GetRandomQuad(std::mt19937& gen, const std::vector<Quad>& quads, float eps) {
    std::uniform_real_distribution<float> err_dist(-eps, eps);
    std::uniform_int_distribution<int> pos_dist(0, quads.empty() ? 0 : quads.size() - 1);

    size_t pos = pos_dist(gen);

    std::array<float, 4> err_array;
    for (int ind = 0; ind < 4; ++ind) {
        err_array[ind] = err_dist(gen);
    }

    const std::array<float, 4> target = quads[pos].GetPoint();
    return Quad(target[0] + err_array[0], target[1] + err_array[1], target[2] + err_array[2], target[3] + err_array[3]);
}

std::vector<Quad> GetSortedQuads(std::vector<Quad> quads, const Quad& target) {
    auto comp = [&target](const Quad& left, const Quad& right) {
        return left.GetDistanceTo(target) < right.GetDistanceTo(target);
    };
    std::sort(quads.begin(), quads.end(), comp);
    return quads;
}

std::unique_ptr<KDTree<Quad>> BuildTreeQuads(std::vector<Quad>&& quads, int leaf_amount) {
    KDTreeBuilder<Quad> builder(SplitType::Median, leaf_amount / 2);
    builder.LoadObjects(std::move(quads));

    return builder.BuildTree();
}

inline void AssertQuads(const std::vector<Quad>& expected, const std::vector<const Quad*>& actual, float esp) {
    EXPECT_NEAR(expected[0].GetPoint()[0], actual[0]->GetPoint()[0], esp);
    EXPECT_NEAR(expected[0].GetPoint()[1], actual[0]->GetPoint()[1], esp);
    EXPECT_NEAR(expected[0].GetPoint()[2], actual[0]->GetPoint()[2], esp);
    EXPECT_NEAR(expected[0].GetPoint()[3], actual[0]->GetPoint()[3], esp);
}

TEST(Quads, Stress) {
    const int tests_amount = 100;
    std::mt19937 gen(343);
    std::uniform_int_distribution<int> amount_dist(100, 20000);
    std::uniform_real_distribution<float> err_dist(0, 10);


    for (int test = 0; test < tests_amount; ++test) {
        const int amount = amount_dist(gen);
        const float err = err_dist(gen);
        auto quads = CreateQuads(gen, amount);
        Quad target = GetRandomQuad(gen, quads, err);

        auto raw_sort_start = std::chrono::steady_clock::now();
        auto sorted_quads = GetSortedQuads(quads, target);
        auto raw_sort_finish = std::chrono::steady_clock::now();
        auto raw_sort_duration = std::chrono::duration_cast<std::chrono::microseconds>(raw_sort_finish - raw_sort_start);

        auto tree_start = std::chrono::steady_clock::now();
        auto tree = BuildTreeQuads(std::move(quads), amount / 2);
        auto tree_build_finish = std::chrono::steady_clock::now();
        auto tree_build_duration = std::chrono::duration_cast<std::chrono::microseconds>(tree_build_finish - tree_start);
        int search_amount = amount < 10 ? amount : 10;

        auto res = tree->SearchClosest(target, err * 2, search_amount);
        auto tree_search_finish = std::chrono::steady_clock::now();
        auto tree_search_duration = std::chrono::duration_cast<std::chrono::microseconds>(tree_search_finish - tree_build_finish);

//        auto raw_sort_duration_us = raw_sort_duration.count();
//        auto tree_build_duration_us = tree_build_duration.count();
//        auto tree_search_duration_us = tree_search_duration.count();
//
//        std::cerr << "Amount:      " << amount << '\n';
//        std::cerr << "Size:      " << sizeof(*tree.get()) << '\n';
//        std::cerr << "Raw sort:    " << raw_sort_duration_us << " us\n";
//        std::cerr << "Tree build:  " << tree_build_duration_us << " us\n";
//        std::cerr << "Tree search: " << tree_search_duration_us << " us\n\n";

//        ASSERT_EQ(res.size(), search_amount);
        AssertQuads(sorted_quads, res, err);
    }
}
