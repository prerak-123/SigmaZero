#include <boost/python.hpp>
#include <Python.h>
#include <boost/multiprecision/cpp_dec_float.hpp>
#include <boost/multiprecision/cpp_int.hpp>
#include <boost/python/numpy.hpp>
#include <boost/scoped_array.hpp>

#include <cstdlib>
#include <vector>
#include <string>
#include <functional>
#include <iostream>
#include <cmath>
#include<unordered_map>

long long int cnt = 0;

#define ll long long int
#define AMOUNT_OF_PLANES 73
#define BOARD_SIZE 8
#define DIRICHLET_NOISE 0.3
#define C_BASE 20000
#define C_INIT 2

typedef boost::python::object P_Action;
typedef boost::python::object P_Agent;
typedef boost::python::object P_List; 
typedef boost::python::object P_Array;
typedef boost::python::object P_Generator;
typedef boost::python::object P_Board;
typedef boost::python::object P_Move;
// #define P_Action boost::python::object
// #define P_Agent boost::python::object
// #define P_List boost::python::object
// #define P_Board boost::python::object
// #define P_Move boost::python::object

class C_Node;
class C_Edge;
class C_MCTS;
class C_Edge;

class C_Node{
    public:
        std::string state;
        int turn;
        std::vector<C_Edge*> edges;
        long long int N;
        long double value;

        C_Node();
        C_Node(std::string state);
        std::string step(P_Action action);
        bool is_game_over();
        bool is_leaf();
        uint64_t add_child(C_Node* child, P_Action action, long double prior);
        std::vector<C_Node*> get_all_children();
        void get_all_children_helper(std::vector<C_Node*> &children);
        uint64_t get_edge(P_Action action);
};

class C_Edge{   
    public:
        C_Node* input_node;
        C_Node* output_node;
        P_Action action;

        bool player_turn;

        uint64_t N;
        uint64_t W;
        long double P;

        C_Edge();
        C_Edge(C_Node* input_node, C_Node* output_node, P_Action action, long double prior);
		long double upper_confidence_bound(long double noise);
        std::string get_action_uci();
        uint64_t get_N();
};



class C_MCTS{
    public:
        C_Node* root;
        std::vector<C_Edge*> game_path;
        P_Board cur_board;
		P_Agent agent;
		bool stochastic;
		std::vector<std::vector<boost::python::object>> outputs;

        void run_simulations(int n);
        uint64_t select_child(C_Node* node);
        void map_valid_move(P_Move move);
        std::unordered_map<std::string, long double> probabilities_to_actions(P_Array& probabilities, std::string board); 
        uint64_t expand(C_Node* leaf);
        uint64_t backpropagate(C_Node* end_node, long double value);
        bool move_root(uint64_t move0, uint64_t move1);
        uint64_t get_sum_N();
        uint64_t get_edge_N(uint64_t edge);
        P_List get_all_edges(P_List lst);
        P_Action get_edge_action(uint64_t edge);
        std::string get_edge_uci(uint64_t edge);

        C_MCTS();
        C_MCTS(P_Agent agent, std::string state, bool stochastic = false);
        ~C_MCTS();
};     


void delete_mcts_tree(C_Node* node){
    for(int i=0; i < node->edges.size(); ++i){
        delete_mcts_tree(node->edges[i]->output_node);
    }

    delete node;
}

C_Node::C_Node(){}

C_Node::C_Node(std::string state){
    this->state = state;
    this->N = 0;
    this->value = 0;
}

std::string C_Node::step(P_Action action){
    boost::python::object chess_module = boost::python::import("chess");
    boost::python::object board = chess_module.attr("Board")(this->state);
    board.attr("push")(action);
    std::string new_state = boost::python::extract<std::string>(board.attr("fen")());
    return new_state;    
}

bool C_Node::is_game_over(){
    // Py_Initialize();
    boost::python::object chess_module = boost::python::import("chess");
    boost::python::object board = chess_module.attr("Board")(this->state);
    bool is_game_over = boost::python::extract<bool>(board.attr("is_game_over")());
    // Py_Finalize();
    return is_game_over;
}

bool C_Node::is_leaf(){
    return this->N == 0;
}

uint64_t C_Node::add_child(C_Node* child, P_Action action, long double prior){
    C_Edge* edge = new C_Edge(this, child, action, prior);
    this->edges.push_back(edge);
    return (uint64_t)edge;
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

uint64_t C_Node::get_edge(P_Action action){
    for (int i = 0; i < this->edges.size(); i++){
        if (boost::python::extract<bool>(action == this->edges[i]->action)){
            return (uint64_t)this->edges[i];
        }
    }	
    return 0;
}

C_Edge::C_Edge(){}

C_Edge::C_Edge(C_Node* input_node, C_Node* output_node, P_Action action, long double prior){
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

std::string C_Edge::get_action_uci(){
    return boost::python::extract<std::string>(this->action.attr("uci")());
}

uint64_t C_Edge::get_N(){
    return this->N;
}

long double C_Edge::upper_confidence_bound(long double noise){

    long double exploration_rate = std::log((long double)(1 + this->input_node->N + C_BASE) / C_BASE) + C_INIT;
    long double ucb = exploration_rate * (this->P * noise) * (std::sqrt((this->input_node->N)) / (1 + this->N));

    if(this->input_node->turn){
        long double ret =  (long double)(this->W) / (this->N + 1) + ucb;
        return ret;
    }
    else{
        long double ret =  -((long double)(this->W) / (this->N + 1)) + ucb;
        return ret;
    }
}

C_MCTS::C_MCTS(){}

C_MCTS::C_MCTS(P_Agent agent, std::string state, bool stochastic){
    this->root = new C_Node(state);
    this->agent = agent;
    this->stochastic = stochastic;
}

C_MCTS::~C_MCTS(){
    delete_mcts_tree(this->root);
}

void C_MCTS::run_simulations(int n){
    for(int i=0; i<n; ++i){
        this->game_path = std::vector<C_Edge*> ();
        C_Node* leaf = (C_Node*)select_child(root);
        leaf->N += 1;
        leaf = (C_Node*)expand(leaf);
        leaf = (C_Node*)backpropagate(leaf, leaf->value);
    }
}

uint64_t C_MCTS::select_child(C_Node* node){
    while(! node->is_leaf()){
        if(! node->edges.size()){
            return (uint64_t)node;
        }

        boost::python::object numpy = boost::python::import("numpy");
        P_Array dirichlet_noise = numpy.attr("ones")(node->edges.size());

        if(this->stochastic && node->state == this->root->state){
            boost::python::object dirichlet = numpy.attr("random").attr("dirichlet");
            P_Array arr = numpy.attr("full")(node->edges.size(), DIRICHLET_NOISE);
            dirichlet_noise = dirichlet(arr);
        }
        // else{
        //     dirichlet_noise = numpy.attr("ones")(node->edges.size());
        // }

        C_Edge* best_edge = NULL;
        long double best_score = -1e10;

        for(int i=0; i<node->edges.size(); i++){
            C_Edge* edge = node->edges[i];
            long double score = edge->upper_confidence_bound(boost::python::extract<long double>(dirichlet_noise[i]));
            if(score > best_score){
                best_edge = edge;
                best_score = score;
            }
        }

        if(best_edge == NULL){
            assert(0);
        }

        this->game_path.push_back(best_edge);

        return (uint64_t)best_edge->output_node;
    }

    return (uint64_t)node;
}

void C_MCTS::map_valid_move(boost::python::object move){
	boost::python::object from_square = move.attr("from_square");
	boost::python::object to_square = move.attr("to_square");
	boost::python::object chess = boost::python::import("chess");
	boost::python::object plane_index;
    boost::python::object piece = this->cur_board.attr("piece_at")(from_square);
	boost::python::object Mapping = boost::python::import("mapper");

    int from = boost::python::extract<int>(from_square);
    int to = boost::python::extract<int>(to_square);

	if(boost::python::extract<bool>(move.attr("promotion")) && boost::python::extract<bool>(move.attr("promotion") != chess.attr("QUEEN"))){
	    boost::python::object x = Mapping.attr("get_underpromotion_move")(boost::python::extract<bool>(move.attr("promotion")), from, to);
		
		plane_index = Mapping.attr("mapper")[x[0]][1-x[1]];
	}
	else{
		if(boost::python::extract<bool>(piece.attr("piece_type") == chess.attr("KNIGHT"))){
			boost::python::object direction = Mapping.attr("get_knight_move")(from, to);
			plane_index = Mapping.attr("mapper")[direction];
		}
		else{
			boost::python::object x = Mapping.attr("get_queenlike_move")(from, to);
			boost::python::object np = boost::python::import("numpy");
            int x0 = boost::python::extract<int>(x[0]);
            int x1 = boost::python::extract<int>(x[1]);
			plane_index = Mapping.attr("mapper")[x0][np.attr("abs")(x1)-1];
		}
	}

	boost::python::object row = from_square % 8;
	boost::python::object col = 7 - (from_square / 8);
	this->outputs.push_back({move, plane_index, row, col});

}

// TODO : Need suggestions of how to implement probabilities to actions

std::unordered_map<std::string, long double> C_MCTS::probabilities_to_actions(P_List& probabilities, std::string bord){
	std::unordered_map <std::string, long double> actions;
	boost::python::object chess = boost::python::import("chess");
	this->cur_board = chess.attr("Board")(bord);
	P_Generator valid_moves = this->cur_board.attr("generate_legal_moves")();

	this->outputs = std::vector<std::vector<boost::python::object>> ();
	boost::python::object num_valid_moves = this->cur_board.attr("legal_moves").attr("count")();
	int num = boost::python::extract<int>(num_valid_moves);

	for(int i=0; i<num; i++){
		boost::python::object move = valid_moves.attr("__next__")();
		this->map_valid_move(move);
	}

	for(int i=0; i < this->outputs.size(); ++i){
		std::string mv = boost::python::extract<std::string>(this->outputs[i][0].attr("uci")());
		int pi = boost::python::extract<int>(this->outputs[i][1]);
		int col = boost::python::extract<int>(this->outputs[i][2]);
		int row = boost::python::extract<int>(this->outputs[i][3]);

		actions[mv] = boost::python::extract<long double>(probabilities[pi*BOARD_SIZE*BOARD_SIZE +  col*BOARD_SIZE + row].attr("item")());
	}

	return actions;
}


uint64_t C_MCTS::expand(C_Node* leaf){
	boost::python::object chess = boost::python::import("chess");
	boost::python::object board = chess.attr("Board")(leaf->state);

	P_Generator possible_actions = board.attr("generate_legal_moves")();
	boost::python::object num_valid_moves = board.attr("legal_moves").attr("count")();
	int num = boost::python::extract<int>(num_valid_moves);

	if(num == 0){
		boost::python::object outcome = board.attr("outcome")(true);

		if(outcome.is_none()){
			leaf->value = 0;
		}
		else{
			if(boost::python::extract<bool>(outcome.attr("winner") == chess.attr("WHITE"))){
				leaf->value = 1;
			}
			else if(boost::python::extract<bool>(outcome.attr("winner") == chess.attr("BLACK"))){
				leaf->value = -1;
			}
			else{
				leaf->value = 0;
			}
		}
		return (uint64_t)leaf;
	}

	boost::python::object ChessEnv = boost::python::import("chessEnv");
	boost::python::object input_state = ChessEnv.attr("state_to_input")(leaf->state);

	boost::python::object probabilities = this->agent.attr("predict")(input_state);
	boost::python::object p = probabilities[0];
	boost::python::object v = probabilities[1];

    P_Array ps = p.attr("numpy")().attr("flatten")();
	std::unordered_map<std::string, long double> actions = probabilities_to_actions(ps, leaf->state);

	long double val = boost::python::extract<long double>(v);
    leaf->value = val;

    
	for(int i=0; i<num; ++i){
		P_Move move = possible_actions.attr("__next__")();
		std::string new_state = leaf->step(move);
		C_Node* child = new C_Node(new_state);
        std::string key = boost::python::extract<std::string>(move.attr("uci")());
        long double val = actions[key];
		uint64_t edge = leaf->add_child(child, move, val);
	}
	return (uint64_t)leaf;
}

uint64_t C_MCTS::backpropagate(C_Node* end_node, long double value){
	for(int i=0; i<this->game_path.size(); ++i){
		this->game_path[i]->input_node->N += 1;
		this->game_path[i]->N += 1;
		this->game_path[i]->W += value;
	}

	return (uint64_t)end_node;
}

bool C_MCTS::move_root(uint64_t move0, uint64_t move1){
    P_Action action0 = ((C_Edge*)move0)->action;
    P_Action action1 = ((C_Edge*)move1)->action;
    std::string uci0 = boost::python::extract<std::string>(action0.attr("uci")());
    std::string uci1 = boost::python::extract<std::string>(action1.attr("uci")());
    
    bool flag = false;
    C_Node* e;
    for(int i=0; i < root->edges.size(); ++i){
        std::string uci = boost::python::extract<std::string>(root->edges[i]->action.attr("uci")());

        if(uci == uci0){
            e = root->edges[i]->output_node;
            flag = true;
            break;
        }
    }

    if(! flag) return false;

    for(int i=0; i < e->edges.size(); ++i){
        std::string uci = boost::python::extract<std::string>(e->edges[i]->action.attr("uci")());

        if(uci == uci1){

            for(int i=0; i<root->edges.size(); ++i){
                if(root->edges[i]->output_node != e){
                    delete_mcts_tree(root->edges[i]->output_node);
                }
            }

            for(int j=0; j<e->edges.size(); ++j){
                if(i != j){
                    delete_mcts_tree(e->edges[j]->output_node);
                }
            }

            root = e->edges[i]->output_node;
            return true;
        }
    }

    delete_mcts_tree(root);
    return false;
}

uint64_t C_MCTS::get_sum_N(){
    uint64_t sum = 0;
    for(int i=0; i < root->edges.size(); ++i){
        sum += root->edges[i]->N;
    }

    return sum;
}

uint64_t C_MCTS::get_edge_N(uint64_t edge){
    return ((C_Edge*)edge)->N;
}

P_List C_MCTS::get_all_edges(P_List lst){
    for(int i=0; i < root->edges.size(); ++i){
        lst.attr("append")((uint64_t)root->edges[i]);
    }

    return lst;
}

P_Action C_MCTS::get_edge_action(uint64_t edge){
    return ((C_Edge*)edge)->action;
}

std::string C_MCTS::get_edge_uci(uint64_t edge){
    return boost::python::extract<std::string>(((C_Edge*)edge)->action.attr("uci")());
}

BOOST_PYTHON_MODULE(CPP_backend)
{
    Py_Initialize();
    boost::python::class_ <C_Node>("Node")
        .def(boost::python::init<>())
        .def(boost::python::init<std::string>())
        .def("step", &C_Node::step)
        .def("is_game_over", &C_Node::is_game_over)
        .def("is_leaf", &C_Node::is_leaf)
        .def("add_child", &C_Node::add_child)
        .def("get_all_children", &C_Node::get_all_children)
        .def("get_edge", &C_Node::get_edge)
        ;

    boost::python::class_ <C_Edge>("Edge")
        .def(boost::python::init<>())
        .def(boost::python::init<C_Node*, C_Node*, P_Action, long double>())
        .def("upper_confidence_bound", &C_Edge::upper_confidence_bound)
        ;

    boost::python::class_ <C_MCTS>("MCTS")
        .def(boost::python::init<>())
        .def(boost::python::init<P_Agent, std::string, bool>())
        .def("run_simulations", &C_MCTS::run_simulations)
        .def("move_root", &C_MCTS::move_root)
        .def("get_sum_N", &C_MCTS::get_sum_N)
        .def("get_edge_N", &C_MCTS::get_edge_N)
        .def("get_all_edges", &C_MCTS::get_all_edges)
        .def("get_edge_action", &C_MCTS::get_edge_action)
        .def("get_edge_uci", &C_MCTS::get_edge_uci)
        ;
    Py_Finalize();
}

