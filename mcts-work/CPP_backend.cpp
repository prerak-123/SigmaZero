#include <boost/python.hpp>
#include <Python.h>
#include <boost/multiprecision/cpp_dec_float.hpp>
#include <boost/multiprecision/cpp_int.hpp>
#include <cstdlib>
#include <vector>
#include <string>
#include <functional>

#define ll long long int

class C_Node;
class C_Edge;
class C_MCTS;
class C_Edge;

class C_Node{
    public:
        std::string state;
        int turn;
        std::vector<C_Edge*> edges;
        boost::multiprecision::cpp_int N;
        boost::multiprecision::cpp_dec_float_50 value;

        C_Node(std::string state);
        std::string step(boost::python::object action);
        bool is_game_over();
        bool is_leaf();
        C_Edge* add_child(C_Node* child, boost::python::object action, boost::multiprecision::cpp_dec_float_50 prior);
        std::vector<C_Node*> get_all_children();
        void get_all_children_helper(std::vector<C_Node*> &children);
        C_Edge* get_edge(boost::python::object action);
};

class C_Edge{   
    public:
        C_Node* input_node;
        C_Node* output_node;
        boost::python::object action;

        bool player_turn;

        boost::multiprecision::cpp_int N;
        boost::multiprecision::cpp_dec_float_50 W;
        boost::multiprecision::cpp_dec_float_50 P;

        C_Edge(C_Node* input_node, C_Node* output_node, boost::python::object action, boost::multiprecision::cpp_dec_float_50 prior);
};

// class C_Action{

// }

class C_MCTS{
    public:
        C_Node* root;
        std::vector<C_Node*> game_path;
        boost::python::object cur_board;
        // TODO: put agent in here somehow
        bool stochastic;

        void run_simulations(int n);
        C_Node* select_child(C_Node* node);
        void map_valid_move(boost::python::object move);
        std::unordered_map<boost::python::object, boost::multiprecision::cpp_dec_float_50> probabilities_to_actions(vector<vector<vector<boost::multiprecision::cpp_dec_float_50>>> probabilities, std::string board); 
        C_Node* expand(C_Node* leaf);
        C_Node* backpropagate(C_Node* end_node, boost::multiprecision::cpp_dec_float_50 value);

        C_MCTS::C_MCTS(boost::python::object agent, std::string state, bool stochastic = false);
};      

C_Node::C_Node(std::string state){
    this->state = state;
    // this->turn = ; // TODO: Insert python function here
    this->N = 0;
    this->value = 0;
}

std::string C_Node::step(boost::python::object action){
    Py_Initialize();
    boost::python::object chess_module = boost::python::import("chess");
    // boost::python::object Move_from_chess = chess_module.attr("Move");
    boost::python::object board = chess_module.attr("Board")(this->state);
    // boost::python::object board = Board_function(this->state);
    boost::python::object new_board = board.attr("push")(action);
    std::string new_state = boost::python::extract<std::string>(new_board.attr("fen")());
    Py_Finalize();
    return new_state;    
}

bool C_Node::is_game_over(){
    Py_Initialize();
    boost::python::object chess_module = boost::python::import("chess");
    boost::python::object board = chess_module.attr("Board")(this->state);
    bool is_game_over = boost::python::extract<bool>(board.attr("is_game_over")());
    Py_Finalize();
    return is_game_over;
}

bool C_Node::is_leaf(){
    return this->N == 0;
}

C_Edge* C_Node::add_child(C_Node* child, boost::python::object action, boost::multiprecision::cpp_dec_float_50 prior){
    C_Edge* edge = new C_Edge(this, child, action, prior);
    this->edges.push_back(edge);
    return edge;
}

std::vector<C_Node*> C_Node::get_all_children(){
    std::vector<C_Node*> children;
    get_all_children_helper(children);
    return children;
}

void C_Node::get_all_children_helper(std::vector<C_Node*> &children){
    for (int i = 0; i < this->edges.size(); i++){
        children.push_back(this->edges[i]->output_node);
        assert(this->edges[i]->output_node != NULL);
        this->edges[i]->output_node->get_all_children_helper(children);
    }
}

C_Edge* C_Node::get_edge(boost::python::object action){
    for (int i = 0; i < this->edges.size(); i++){
        if (action == this->edges[i]->action){
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

    return NULL;
}

C_Edge::C_Edge(C_Node* input_node, C_Node* output_node, boost::python::object action, boost::multiprecision::cpp_dec_float_50 prior){
    this->input_node = input_node;
    this->output_node = output_node;
    this->action = action;

    std::string temp = input_node->state;
    int end = temp.find(" ");
    temp.erase(temp.begin(), temp.begin() + end + 1);
    end = temp.find(" ");
    this->player_turn = temp.substr(0, end) == "w";

    this->N = 0;
    this->W = 0;
    this->P = prior;
}

boost::multiprecision::cpp_dec_float_50 C_Edge::upper_confidence_bound(boost::multiprecision::cpp_dec_float_50 noise){
    // TODO : Implement this shit
    return 0;
}

C_MCTS::C_MCTS(boost::python::object agent, std::string state, bool stochastic = false){
    this->root = new C_Node(state);
    this->agent = agent;
    this->stochastic = stochastic;
}

// int main(){
//     std::cout << "hello" << std::endl;
// }