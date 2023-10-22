#include "../inc/mcts.h"

#include "state_action.cpp"

using namespace boost::python;

TreeNode::TreeNode(BoardState state, TreeNode *parent)
{
    this->state = state;
    this->parent = parent;
    this->isTerminal = state.isTerminal();
    this->isFullyExpanded = this->isTerminal;
    this->numVisits = 0;
    this->totalReward = 0;
}

MCTS::MCTS()
{
}

MCTS::MCTS(std::string limitType, cpp_int limit, cpp_dec_float_50 explorationConstant, std::function<cpp_dec_float_50(BoardState)> rollout)
{
}

Action MCTS::search(BoardState rootState)
{
    Action a;
    return a;
}

void MCTS::executeRound()
{
}

TreeNode *MCTS::expand(TreeNode *node)
{
    return nullptr;
}

void MCTS::backPropagate(TreeNode *node, cpp_dec_float_50 reward)
{
}

TreeNode *MCTS::getBestChild(TreeNode *node, cpp_dec_float_50 explorationValue)
{
    return nullptr;
}

BOOST_PYTHON_MODULE(MCTS)
{   
    class_<MCTS>("MCTS")
        .def(init<>())
        .def(init<std::string, cpp_int, cpp_dec_float_50, std::function<cpp_dec_float_50(BoardState)>>())
        .def("search", &MCTS::search)
        .def("executeRound", &MCTS::executeRound)
        .def("expand", &MCTS::expand)
        .def("backPropagate", &MCTS::backPropagate)
        .def("getBestChild", &MCTS::getBestChild)
    ;
}
