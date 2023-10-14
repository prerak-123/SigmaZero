#include <boost/python.hpp>
#include <boost/multiprecision/cpp_dec_float.hpp>
#include <boost/multiprecision/cpp_int.hpp>
#include <cstdlib>
#include <vector>
#include <string>
#include <functional>

#include "../inc/state_action.h"

using boost::multiprecision::cpp_dec_float_50;
using boost::multiprecision::cpp_int;

class TreeNode{
    private:
        BoardState state;
        bool isTerminal;
        bool isFullyExpanded;
        TreeNode* parent;
        cpp_int numVisits;
        cpp_dec_float_50 totalReward;
        std::vector<TreeNode*> children;

    public:
        TreeNode(BoardState state, TreeNode* parent);
};

class MCTS{
    private:
        std::string limitType;
        cpp_int timeLimit; // in milliseconds
        cpp_int iterationLimit; // count of iterations
        cpp_dec_float_50 explorationConstant;
        cpp_dec_float_50 (*rollout)(BoardState);

    public:
        MCTS();
        MCTS(std::string limitType, cpp_int limit, cpp_dec_float_50 explorationConstant, std::function<cpp_dec_float_50(BoardState)> rollout);
        Action search(BoardState rootState);
        void executeRound();
        TreeNode* expand(TreeNode* node);
        void backPropagate(TreeNode* node, cpp_dec_float_50 reward);
        TreeNode* getBestChild(TreeNode* node, cpp_dec_float_50 explorationValue);
};