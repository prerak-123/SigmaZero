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

class C_MCTS{
    public:
        C_Node* root;
        std::vector<Node*> game_path;
        std::string cur_board;
        // TODO: put agent in here somehow
        bool stochastic;

        run_simulations(int n);
        C_Node* select_child(C_Node* node);
        void map_valid_move(/*TODO somthing here*/);    
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

uint64_t C_Node_Alloter(std::string state){
    C_Node* node = new C_Node(state);
    return (uint64_t) node;
}

C_Node* select_child(C_Node* node){
    while(! node->is_leaf()){
        if(! node->edges.size()){
            return node;
        }

        std::vector<boost::multiprecision::cpp_dec_float> noise (node->edges.size(), 1);

        if(this->stochastic && node->state == this->root->state){
            // TODO : Put dirichlet noise thing
        }

        C_Edge* best_edge = NULL;
        boost::multiprecision::cpp_dec_float best_score = -1e10;

        for(int i=0; i<node->edges.size(); i++){
            C_Edge* edge = node->edges[i];
            boost::multiprecision::cpp_dec_float score = edge->upper_confidence_bound(noise[i]);
            if(score > best_score){
                best_edge = edge;
                best_score = score;
            }
        }

        if(best_edge == NULL){
            assert(0);
        }

        this->game_path.push_back(best_edge);

        return best_edge->output_node;
    }
}

