// #include "../inc/state_action.h"

using namespace boost::python;

// g++ state_action.cpp -shared -fpic -Wno-undef -I /usr/include/python3.10/ -L /usr/lib/x86_64-linux-gnu/  -lboost_python310 -lboost_system -o StateAction.so

BoardState::BoardState(){
    board = "";
    current_player = 1;
}

BoardState::BoardState(std::string board, int current_player){
    this->board = board;
    this->current_player = current_player;
}

int BoardState::getCurrentPlayer(){
    return current_player;
}

std::vector<Action> BoardState::getPossibleActions(){
    std::vector<Action> possible_actions;
    // TODO: Implement getPossibleActions
    return possible_actions;
}

BoardState BoardState::takeAction(Action action){
    BoardState new_state;
    // TODO: Implement takeAction
    return new_state;
}

bool BoardState::isTerminal(){
    // TODO: Implement isTerminal
    return false;
}

cpp_dec_float_50 BoardState::getReward(){
    // TODO: Implement getReward
    return 0;
}

BOOST_PYTHON_MODULE(StateAction)
{
    class_<BoardState>("BoardState")
        .def(init<>())
        .def("getCurrentPlayer", &BoardState::getCurrentPlayer)
        .def("getPossibleActions", &BoardState::getPossibleActions)
        .def("takeAction", &BoardState::takeAction)
        .def("isTerminal", &BoardState::isTerminal)
        .def("getReward", &BoardState::getReward)
    ;
}