#ifndef KDTREE_KD_TREE_H
#define KDTREE_KD_TREE_H

#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <memory>
#include <queue>
#include <string>
#include <vector>

const float INITIAL_LIMIT = INT64_MAX;
template <int Dimension>
class KDTreePlaceable {
public:
    virtual std::array<float, Dimension> GetPointMax() const = 0;
    virtual std::array<float, Dimension> GetPointMin() const = 0;
    virtual float GetDistanceTo(const KDTreePlaceable<Dimension>& object) const = 0;

    constexpr static int GetDimension() {
        return Dimension;
    }
};

template<class Object>
class KDTreeBuilder;

class SurfaceAreaHeuristic {
public:
    SurfaceAreaHeuristic(float ci = 1, float ct = 2) : ci_(ci), ct_(ct) {
    }

    // f(x)
    float operator()(float SAL, float SAR, float SAP, int NL, int NR) const {
        return (ct_ + ci_ * ((SAL * NL + SAR * NR) / (SAP)));
    }

    // f_0
    float operator()(int N) const {
        return ci_ * N;
    }

private:
    float ci_;
    float ct_;
};

template<class Object>
class KDTree {
public:
    KDTree() = delete;

    KDTree(KDTree&& tree)
    : objects_(std::move(tree.objects_)), root_(tree.root_), dim_(tree.dim_), k_(tree.k_) {
    }

    bool SaveToFile(const std::string& filename) const;

    const std::vector<Object*> SearchClosest(const Object& obj,
                                             const std::array<float, Object::GetDimension()>& eps) const {
        std::queue<TreeNode*> queue;
        queue.push(root_.get());

        std::vector<Object*> result;
        std::map<size_t, bool> registered_nodes;

        while(!queue.empty()) {
            TreeNode* current_node = queue.front();

            if (current_node->IsLeaf()) {
                for (size_t ind : current_node->objects_) {
                    if (registered_nodes.find(ind) == registered_nodes.end()) {
                        result.push_back(&(objects_[ind]));
                        registered_nodes[ind] = true;
                    }
                }
            } else {
                const TreeNode* left = current_node->GetLeft();
                const TreeNode* right = current_node->GetRight();

                if (left->IsCovering(obj, eps)) {
                    queue.push(left);
                }

                if (right->IsCovering(obj, eps)) {
                    queue.push(right);
                }
            }

            queue.pop();
        }

        if (result.size() <= k_) {
            return result;
        }

        auto comp = [&obj](const Object* left, const Object* right) {
            return left->DistanceTo(obj) < right->DistanceTo(obj);
        };
        std::sort(result.begin(), result.end(), comp);

        return result.resize(k_);
    }

    void SetK(int k) const;

private:
    struct TreeNode {
        TreeNode(size_t split_dimension,
                 const std::array<float, Object::GetDimension()>& point_max,
                 const std::array<float, Object::GetDimension()>& point_min,
                 std::vector<size_t>&& objects) :
                 split_dimension_(split_dimension),
                 left_(nullptr), right_(nullptr),
                 point_max_(point_max), point_min_(point_min),
                 objects_(std::move(objects)) {
        }

        bool IsLeaf() const {
            return ((left_ == nullptr) && (right_ == nullptr));
        }

        size_t GetObjectsAmount() const {
            return (objects_.size());
        }

        float GetSpace() const {
            float space = 1;
            for (int dim = 0; dim < Object::GetDimension(); ++dim) {
                space *= std::abs(point_max_[dim] - point_min_[dim]);
            }

            return space;
        }

        const TreeNode* GetLeft() const {
            return left_.get();
        }

        const TreeNode* GetRight() const {
            return right_.get();
        }

        bool IsCovering(const Object& obj,
                        const std::array<float, Object::GetDimension()>& eps) const {
            const std::array<float, Object::GetDimension()> point_max = obj.GetPointMax();
            const std::array<float, Object::GetDimension()> point_min = obj.GetPointMin();
            for (size_t dim = 0; dim < Object::GetDimension(); ++dim) {
                if (point_min[dim] < (point_min_[dim] - eps[dim])
                || point_max[dim] > (point_max_[dim] + eps[dim])) {
                    return false;
                }
            }
            return true;
        }

        size_t split_dimension_;

        std::unique_ptr<TreeNode> left_;
        std::unique_ptr<TreeNode> right_;

        const std::array<float, Object::GetDimension()> point_max_;
        const std::array<float, Object::GetDimension()> point_min_;

        std::vector<size_t> objects_;
    };

private:
    KDTree(std::vector<TreeNode>&& objects)
    : objects_(std::move(objects)), root_(std::make_unique<TreeNode>()), dim_(Object::GetDimension()), k_(3) {
        root_->split_dimension_ = 0;

        std::array<float, Object::GetDimension()> root_point_max;
        std::array<float, Object::GetDimension()> root_point_min;
        root_point_max.fill(-INITIAL_LIMIT);
        root_point_min.fill(INITIAL_LIMIT);

        for (const Object& obj : objects_) {
            std::array<float, Object::GetDimension()> point_max = obj.GetPointMax();
            std::array<float, Object::GetDimension()> point_min = obj.GetPointMin();

            for (size_t d = 0; d < dim_; ++d) {
                if (root_point_max[d] < point_max[d]) {
                    root_point_max[d] = point_max[d];
                }
                if (root_point_min[d] > point_min[d]) {
                    root_point_min[d] = point_min[d];
                }
            }
        }

        root_->point_max_ = root_point_max;
        root_->point_min_ = root_point_min;

        root_->objects_.reserve(objects_.size());
        for (size_t obj = 0; obj < objects_.size(); ++obj) {
            root_->objects_.push_back(obj);
        }
    }

    std::pair<TreeNode*, TreeNode*> SliceLeaf(TreeNode* leaf, const SurfaceAreaHeuristic& sah_func) {
        std::vector<int> a_high(sections_amount_);
        std::vector<int> a_low(sections_amount_);

        const float max_limit = leaf->point_max_[leaf->split_dimension_];
        const float min_limit = leaf->point_min_[leaf->split_dimension_];
        const float step = (max_limit - min_limit) / sections_amount_;

        for (size_t obj : leaf->objects_) {
            float coord_max = objects_[obj].GetPointMax()[leaf->split_dimension_];
            float coord_min = objects_[obj].GetPointMin()[leaf->split_dimension_];

            for (size_t section = 0; section < sections_amount_; ++section) {
                if ((coord_max > (min_limit + static_cast<float>(section * step)))
                &&  (coord_max < (min_limit + static_cast<float>(section * (step + 1))))) {
                    a_high[section] += 1;
                }

                if ((coord_min > (min_limit + static_cast<float>(section * step)))
                &&  (coord_min < (min_limit + static_cast<float>(section * (step + 1))))) {
                    a_low[section] += 1;
                }
            }
        }

        for (size_t section = 1; section < sections_amount_; ++section) {
            a_high[sections_amount_ - 1 - section] += a_high[sections_amount_ - section];
            a_low[section] += a_low[section - 1];
        }

        // Surface Area Heuristic time!
        // Search for the most suitable area split
        float space_left = 0;
        float space_right = max_limit - min_limit;
        const float total_space = space_right;

        int min_pos = 1;
        float min_val = INITIAL_LIMIT;
        bool is_worth_splitting = false;
        for (int pos = 1; pos < sections_amount_; ++pos) {
            space_left += step;
            space_right -= step;

            // This value is equal to the amount of elements which are present on the both sides
            const int elements_everywhere = leaf->objects_.size() - a_low[pos] + a_high[pos - 1];

            const float val = sah_func(space_left, space_right, total_space, a_high[pos - 1] + elements_everywhere,
                                                                        a_low[pos] + elements_everywhere);

            if (sah_func(leaf->objects_.size()) >= val) {
                is_worth_splitting = true;
            }

            if (val < min_val) {
                min_pos = pos;
                min_val = val;
            }
        }
        if (!is_worth_splitting) {
            return {nullptr, nullptr};
        }

        const float split_limit = min_limit + step * min_pos;

        // Once it is found objects must be split
        const int elements_everywhere = leaf->objects_.size() - a_low[min_pos] + a_high[min_pos - 1];
        std::vector<size_t> left_objects;
        std::vector<size_t> right_objects;
        left_objects.reserve((a_high[min_pos - 1] + elements_everywhere));
        right_objects.reserve(a_low[min_pos] + elements_everywhere);

        for (size_t obj : leaf->objects_) {
            const float coord_max = objects_[obj].GetPointMax()[leaf->split_dimension_];
            const float coord_min = objects_[obj].GetPointMin()[leaf->split_dimension_];

            if (coord_max < split_limit) {
                // left side
                left_objects.push_back(obj);
            } else if (coord_min > split_limit) {
                // right_side
                right_objects.push_back(obj);
            } else {
                // on the edge
                left_objects.push_back(obj);
                right_objects.push_back(obj);
            }
        }

        std::array<float, Object::GetDimension()> left_max = leaf->point_max_;
        std::array<float, Object::GetDimension()> left_min = leaf->point_min_;
        left_max[leaf->split_dimension_] = split_limit;

        std::array<float, Object::GetDimension()> right_max = leaf->point_max_;
        std::array<float, Object::GetDimension()> right_min = leaf->point_min_;
        right_min[leaf->split_dimension_] = split_limit;

        leaf->left_ = std::make_unique<TreeNode>((leaf->split_dimension_ + 1) % dim_, left_max, left_min, std::move(left_objects));
        leaf->right_ = std::make_unique<TreeNode>((leaf->split_dimension_ + 1) % dim_, right_max, right_min, std::move(right_objects));
        leaf->objects_.clear();

        return {leaf->left_.get(), leaf->right_.get()};
    }

    TreeNode* GetRoot() const {
        return root_.get();
    }



private:
    friend class KDTreeBuilder<Object>;

private:
    const std::vector<Object> objects_;
    std::unique_ptr<TreeNode> root_;
    size_t dim_;
    int k_;

    const int sections_amount_ = 32;
};

template<class Object>
class KDTreeBuilder {
public:
    KDTreeBuilder() : sah_func_(), tree_(nullptr) {
    }

    void LoadObjects(std::vector<Object>&& objects) {
        tree_ = KDTree<Object>(std::move(objects));
    }

    std::unique_ptr<KDTree<Object>>&& BuildTree() {
        std::queue<class KDTree<Object>::TreeNode *> queue;
        queue.push(tree_.root_.get());

        while(!queue.empty()) {
            class KDTree<Object>::TreeNode* node = queue.back();
            std::pair<class KDTree<Object>::TreeNode*, class KDTree<Object>::TreeNode*> new_nodes =
                    tree_.SliceLeaf(node, sah_func_);

            // a bit excessive but just in case
            if ((new_nodes.first != nullptr) && (new_nodes.second != nullptr)) {
                queue.push(new_nodes.first);
                queue.push(new_nodes.second);
            }

            queue.pop();
        }



        return std::move(tree_);
    }
    
    KDTree<Object>&& LoadFromFile(const std::string& filename);

private:

    const SurfaceAreaHeuristic sah_func_;
    std::unique_ptr<KDTree<Object>> tree_;
};

#endif //KDTREE_KD_TREE_H
