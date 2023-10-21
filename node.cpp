#include <boost/python.hpp>
#include <boost/multiprecision/cpp_dec_float.hpp>
#include <boost/multiprecision/cpp_int.hpp>
#include <cstdlib>
#include <vector>
#include <string>
#include <functional>

#define ll long long int

class C_Node{
    public:
        std::string state;
        int turn;
        std::vector<C_Edge*> edges;
        boost::multiprecision::cpp_int N;
        boost::multiprecision::cpp_dec_float value;

        C_Node(std::string state);
        std::string step(C_Action* action);
        bool is_game_over();
        bool is_leaf();
}

class C_Edge{   
    public:
        C_Node* input_node;
        C_Node* output_node;
        C_Action* action;

        bool player_turn;

        boost::multiprecision::cpp_int N;
        boost::multiprecision::cpp_dec_float W;
        boost::multiprecision::cpp_dec_float P;


};

class C_Action{

}


C_Node::C_Node(std::string state){
    this->state = state;
    this->turn = ; // TODO: Insert python function here
    this->N = 0;
    this->value = 0;
}

std::string C_Node::step(C_Action* action){
    std::string new_state = ; // TODO: Insert python function here
    return new_state;    
}

bool C_Node::is_game_over(){
    bool is_game_over = ; // TODO: Insert python function here
    return is_game_over;
}

bool C_Node::is_leaf(){
    return this->N == 0;
}

Edge* C_Node::add_child(C_Node* child, Action* action, boost::multiprecision::cpp_dec_float prior){
    Edge* edge = new Edge(this, child, prior);
    this->edges.push_back(edge);
    return edge;
}

std::vector<Node*> C_Node::get_all_children(){
    std::vector<Node*> children;
    get_all_children_helper(children);
    return children;

    return children;
}

void C_Node::get_all_children_helper(std::vector<Node*> &children){
    for (int i = 0; i < this->edges.size(); i++){
        children.push_back(this->edges[i]->output_node);
        assert(this->edges[i]->output_node != NULL);
        this->edges[i]->output_node->get_all_children_helper(children);
    }
}

Edge* C_Node::get_edge(Action* action){
    for (int i = 0; i < this->edges.size(); i++){
        if (/*Implement equality of C_Action and put it here*/){
            return this->edges[i];
        }
    }
    return NULL;
}
