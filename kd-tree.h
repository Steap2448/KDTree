#ifndef KDTREE_KD_TREE_H
#define KDTREE_KD_TREE_H

#include <array>
#include <string>
#include <vector>

template <int Dimension>
class KDTreePlaceable {
public:
    virtual std::array<double, Dimension> GetPointMax() const = 0;
    virtual std::array<double, Dimension> GetPointMin() const = 0;

    constexpr static int GetDimension() {
        return Dimension;
    }
};

template<class Object>
class KDTreeBuilder;

template<class Object>
class KDTree {
public:
    KDTree() = delete;

    bool SaveToFile(const std::string& filename) const;
    const std::vector<Object*> SearchClosest(const Object& obj) const;
    void SetK(int k) const;

private:
    struct TreeNode {
        bool IsLeaf() const;

        TreeNode* left_;
        TreeNode* right_;

        const std::array<double, Object::GetDimension()> point_max_;
        const std::array<double, Object::GetDimension()> point_min_;

        std::vector<Object*> objects_;
    };

private:
    std::pair<TreeNode*, TreeNode*> SliceLeaf(TreeNode* leaf);
    TreeNode* GetRoot() const;

private:
    friend class KDTreeBuilder<Object>;

private:
    std::vector<Object> objects_;
    std::vector<TreeNode> nodes_;
    const TreeNode& root_;
    int dim_;
};

class SurfaceAreaHeuristic {
public:
    SurfaceAreaHeuristic(double ci, double ct) : ci_(ci), ct_(ct) {
    }

    double operator()(double SAL, double SAR, double SAP, int NL, int NR) const;

private:
    double ci_;
    double ct_;
};

template<class Object>
class KDTreeBuilder {
public:
    void LoadObjects(std::vector<Object>&& objects);
    KDTree<Object>&& BuildTree();
    
    KDTree<Object>&& LoadFromFile(const std::string& filename);

private:
    const SurfaceAreaHeuristic sah_func_;
    const int sections_amount_;

    KDTree<Object> tree_;
};

#endif //KDTREE_KD_TREE_H
