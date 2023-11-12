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
//TODO: 1. implement change node function -- DONE
//TODO: 2. implement functionalities for getting edge action ucis and N values -- DONE
//TODO: 3. Implement remaining functions

#define ll long long int
#define AMOUNT_OF_PLANES 73
#define BOARD_SIZE 8
#define DIRICHLET_NOISE 0.3
#define C_BASE 20000
#define C_INIT 2

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

        C_Node();
        C_Node(std::string state);
        std::string step(boost::python::object action);
        bool is_game_over();
        bool is_leaf();
        uint64_t add_child(C_Node* child, boost::python::object action, double prior);
        std::vector<C_Node*> get_all_children();
        void get_all_children_helper(std::vector<C_Node*> &children);
        uint64_t get_edge(boost::python::object action);
};

class C_Edge{   
    public:
        C_Node* input_node;
        C_Node* output_node;
        boost::python::object action;

        bool player_turn;

        uint64_t N;
        // boost::multiprecision::cpp_dec_float_50 W;
        // boost::multiprecision::cpp_dec_float_50 P;
        double W;
        double P;

        C_Edge();
        C_Edge(C_Node* input_node, C_Node* output_node, boost::python::object action, double prior);
		boost::multiprecision::cpp_dec_float_50 upper_confidence_bound(double noise);
        std::string get_action_uci();
        uint64_t get_N();
};

// class C_Action{

// }

class C_MCTS{
    public:
        C_Node* root;
        std::vector<C_Edge*> game_path;
        boost::python::object cur_board;
        // TODO: put agent in here somehow
		boost::python::object agent;
		bool stochastic;
		std::vector<std::vector<boost::python::object>> outputs;

        void run_simulations(int n);
        uint64_t select_child(C_Node* node);
        void map_valid_move(boost::python::object move);
        std::unordered_map<std::string, double> probabilities_to_actions(boost::python::object probabilities, std::string board); 
        uint64_t expand(C_Node* leaf);
        uint64_t backpropagate(C_Node* end_node, boost::multiprecision::cpp_dec_float_50 value);
        bool move_root(uint64_t move0, uint64_t move1);
        uint64_t get_sum_N();
        uint64_t get_edge_N(uint64_t edge);
        boost::python::object get_all_edges(boost::python::object lst);
        boost::python::object get_edge_action(uint64_t edge);
        std::string get_edge_uci(uint64_t edge);

        C_MCTS();
        C_MCTS(boost::python::object agent, std::string state, bool stochastic = false);
};     

C_Node::C_Node(){}

C_Node::C_Node(std::string state){
    this->state = state;
    // this->turn = ; // TODO: Insert python function here
    this->N = 0;
    this->value = 0;
}

std::string C_Node::step(boost::python::object action){
    // Py_Initialize();
    // std::cout << "in step" << std::endl;
    boost::python::object chess_module = boost::python::import("chess");
    // boost::python::object Move_from_chess = chess_module.attr("Move");
    boost::python::object board = chess_module.attr("Board")(this->state);
    // boost::python::object board = Board_function(this->state);
    board.attr("push")(action);
    std::string new_state = boost::python::extract<std::string>(board.attr("fen")());
    // Py_Finalize();
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

uint64_t C_Node::add_child(C_Node* child, boost::python::object action, double prior){
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

uint64_t C_Node::get_edge(boost::python::object action){
    for (int i = 0; i < this->edges.size(); i++){
        if (action == this->edges[i]->action){
            return (uint64_t)this->edges[i];
        }
    }	
    return 0;
}

uint64_t C_Node_Alloter(std::string state){
    C_Node* node = new C_Node(state);
    return (uint64_t) node;
}

C_Edge::C_Edge(){}

C_Edge::C_Edge(C_Node* input_node, C_Node* output_node, boost::python::object action, double prior){
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

boost::multiprecision::cpp_dec_float_50 C_Edge::upper_confidence_bound(double noise){
    // TODO : Implement this shit
    boost::python::object math = boost::python::import("math");
    boost::python::object log = math.attr("log");
    boost::python::object sqrt = math.attr("sqrt");

    std::cout << "in upper conf bound" << std::endl;

    boost::multiprecision::cpp_dec_float_50 exploration_rate = boost::python::extract<double>(log((double)(1 + this->input_node->N + C_BASE) / C_BASE)) + C_INIT;
    std::cout << "ishdvjkjeqbrvhjrebhujv0" << std::endl;
    boost::multiprecision::cpp_dec_float_50 ucb = exploration_rate * (this->P * noise) * (boost::python::extract<double>(sqrt((this->input_node->N))) / (1 + this->N));

    std::cout << "1 in upper conf bound" << std::endl;

    if(this->input_node->turn){
        boost::multiprecision::cpp_dec_float_50 ret =  (double)(this->W) / (this->N + 1) + ucb;
        return ret;
    }
    else{
        boost::multiprecision::cpp_dec_float_50 ret =  -((double)(this->W) / (this->N + 1)) + ucb;
        return ret;
    }
}

C_MCTS::C_MCTS(){}

C_MCTS::C_MCTS(boost::python::object agent, std::string state, bool stochastic){
    this->root = new C_Node(state);
    this->agent = agent;
    this->stochastic = stochastic;
    // std::cout << "successfulyy constructed" << std::endl;
}

void C_MCTS::run_simulations(int n){
    std::cout << "running sims" << std::endl;
    for(int i=0; i<n; ++i){
        this->game_path = std::vector<C_Edge*> ();
        // std::cout << "1" << std::endl;
        C_Node* leaf = (C_Node*)select_child(root);
        // std::cout << "2" << std::endl;
        // std::cout << leaf << std::endl;
        leaf->N += 1;
        // std::cout << "3" << std::endl;
        leaf = (C_Node*)expand(leaf);
        // std::cout << "4" << std::endl;
        leaf = (C_Node*)backpropagate(leaf, leaf->value);
        // std::cout << "5" << std::endl;
    }
}

uint64_t C_MCTS::select_child(C_Node* node){
    while(! node->is_leaf()){
        // std::cout << "in while" << std::endl;
        if(! node->edges.size()){
            return (uint64_t)node;
        }

        // std::vector<boost::multiprecision::cpp_dec_float_50> noise (node->edges.size(), 1);
        boost::python::object numpy = boost::python::import("numpy");
        boost::python::object dirichlet_noise = numpy.attr("ones")(node->edges.size());

        if(this->stochastic && node->state == this->root->state){
            // TODO : Put dirichlet noise thing
            std::cout << "before dirichlet" << std::endl;
            boost::python::object dirichlet = numpy.attr("random").attr("dirichlet");
            std::cout << "2 before dirichlet" << std::endl;
            boost::python::object arr = numpy.attr("full")(node->edges.size(), DIRICHLET_NOISE);
            dirichlet_noise = dirichlet(arr);
            std::cout << "3 before dirichlet" << std::endl;
        }

        C_Edge* best_edge = NULL;
        boost::multiprecision::cpp_dec_float_50 best_score = -1e10;

        for(int i=0; i<node->edges.size(); i++){
            C_Edge* edge = node->edges[i];
            std::cout << "in for" << std::endl;
            boost::multiprecision::cpp_dec_float_50 score = edge->upper_confidence_bound(boost::python::extract<double>(dirichlet_noise[i]));
            std::cout << "in for 2" << std::endl;
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

    // std::cout << "returning null" << std::endl;
    return (uint64_t)node;
}

void C_MCTS::map_valid_move(boost::python::object move){
	boost::python::object from_square = move.attr("from_square");
	boost::python::object to_square = move.attr("to_square");
	boost::python::object chess = boost::python::import("chess");
	boost::python::object plane_index;
    boost::python::object piece = this->cur_board.attr("piece_at")(from_square);
	// boost::python::object mapper = boost::python::import("mapper");
	// boost::python::object Mapping = mapper.attr("Mapping");
	boost::python::object Mapping = boost::python::import("mapper");

    // std::cout << "in map valid move" << std::endl;

    int from = boost::python::extract<int>(from_square);
    int to = boost::python::extract<int>(to_square);

	if(boost::python::extract<bool>(move.attr("promotion")) && boost::python::extract<bool>(move.attr("promotion") != chess.attr("QUEEN"))){
        // std::cout << "if" << std::endl;
	    boost::python::object x = Mapping.attr("get_underpromotion_move")(move.attr("promotion"), from, to);
		
		plane_index = Mapping.attr("mapper")[x[0]][1-x[1]];
	}
	else{
        // std::cout << "else" << std::endl;
		if(boost::python::extract<bool>(piece.attr("piece_type") == chess.attr("KNIGHT"))){
            // std::cout << from << " " << to << std::endl;
			boost::python::object direction = Mapping.attr("get_knight_move")(from, to);
            // std::cout << "direction " << boost::python::extract<int>(direction) << std::endl;
			plane_index = Mapping.attr("mapper")[direction];
		}
		else{
            // std::cout << "1 83y5783465" << std::endl;
			boost::python::object x = Mapping.attr("get_queenlike_move")(from, to);
            // std::cout << "2 83y5783465" << std::endl;
			boost::python::object np = boost::python::import("numpy");
            // std::cout << "3 83y5783465" << std::endl;
            int x0 = boost::python::extract<int>(x[0]);
            int x1 = boost::python::extract<int>(x[1]);
			plane_index = Mapping.attr("mapper")[x0][np.attr("abs")(x1)-1];
            // std::cout << "4 83y5783465" << std::endl;
		}
	}

    // std::cout << "in map valid move 2" << std::endl;
	
	boost::python::object row = from_square % 8;
	boost::python::object col = 7 - (from_square / 8);
	// TODO: Ensure correctness here
	this->outputs.push_back({move, plane_index, row, col});

    // std::cout << "returning map valid move" << std::endl;
}

// TODO : Need suggestions of how to implement probabilities to actions

std::unordered_map<std::string, double> C_MCTS::probabilities_to_actions(boost::python::object probabilities, std::string bord){
	std::unordered_map <std::string, double> actions;
	boost::python::object chess = boost::python::import("chess");
	this->cur_board = chess.attr("Board")(bord);
	boost::python::object valid_moves = this->cur_board.attr("generate_legal_moves")();

    // double* numpy_probabilities = reinterpret_cast<double*>(probabilities.get_data());
	
	this->outputs = std::vector<std::vector<boost::python::object>> ();
	boost::python::object num_valid_moves = this->cur_board.attr("legal_moves").attr("count")();
	int num = boost::python::extract<int>(num_valid_moves);

    // std::cout << num << std::endl;

	for(int i=0; i<num; i++){
		boost::python::object move = valid_moves.attr("__next__")();
        // std::cout << "about to enter map valid move" << std::endl;
		this->map_valid_move(move);
	}

    // std::cout << "in probabilities to actions" << std::endl;

	for(int i=0; i < this->outputs.size(); ++i){
		std::string mv = boost::python::extract<std::string>(this->outputs[i][0].attr("uci")());
        // std::cout << mv << std::endl;
		int pi = boost::python::extract<int>(this->outputs[i][1]);
        // std::cout << pi << std::endl;
		int col = boost::python::extract<int>(this->outputs[i][2]);
        // std::cout << col << std::endl;
		int row = boost::python::extract<int>(this->outputs[i][3]);
        // std::cout << row << std::endl;

		actions[mv] = boost::python::extract<double>(probabilities[pi*BOARD_SIZE*BOARD_SIZE +  col*BOARD_SIZE + row].attr("item")());
	}

    // std::cout << "returning probabilities to actions" << std::endl;
	return actions;
}


uint64_t C_MCTS::expand(C_Node* leaf){
	// boost::python::object chess = boost::python::import("chess");
	// boost::python::object board = chess.attr("Board")(leaf->state);

	// boost::python::object possible_actions = boost::python::extract<list>(board.attr("generate_legal_moves")());2
	boost::python::object chess = boost::python::import("chess");
	boost::python::object board = chess.attr("Board")(leaf->state);

	boost::python::object possible_actions = board.attr("generate_legal_moves")();
	boost::python::object num_valid_moves = board.attr("legal_moves").attr("count")();
	int num = boost::python::extract<int>(num_valid_moves);

	if(num == 0){
		boost::python::object outcome = board.attr("outcome")(true);

		if(outcome.is_none()){
			leaf->value = 0;
		}
		else{
			if(outcome.attr("winner") == chess.attr("WHITE")){
				leaf->value = 1;
			}
			else if(outcome.attr("winner") == chess.attr("BLACK")){
				leaf->value = -1;
			}
			else{
				leaf->value = 0;
			}
		}
		return (uint64_t)leaf;
	}

    // std::cout << "1 in expand" << std::endl;

	boost::python::object ChessEnv = boost::python::import("chessEnv");
    // std::cout << "2 in expand" << std::endl;
    // std::cout << leaf->state << std::endl;
	boost::python::object input_state = ChessEnv.attr("state_to_input")(leaf->state);

    // std::cout << "3 in expand" << std::endl;

	boost::python::object probabilities = this->agent.attr("predict")(input_state);
    // std::cout << "4 in expand" << std::endl;
	boost::python::object p = probabilities[0];
	boost::python::object v = probabilities[1];
    // std::cout << "5 in expand" << std::endl;

	// TODO: Probability matrix won't work here

    // std::unordered_map<std::string, boost::python::object> actions;
	std::unordered_map<std::string, double> actions = probabilities_to_actions(p.attr("numpy")().attr("flatten")(), leaf->state);
    // std::cout << "6 in expand" << std::endl;

	long double val = boost::python::extract<long double>(v);
    leaf->value = val;

    // std::cout << "7 in expand" << std::endl;
    
    // std::cout << "num " << num << std::endl;
	for(int i=0; i<num; ++i){
        // std::cout << "in for #" << i << std::endl;
		boost::python::object move = possible_actions.attr("__next__")();
        // std::cout << "in for *1" << std::endl;
		std::string new_state = leaf->step(move);
        // std::cout << "in for *2" << std::endl;
		C_Node* child = new C_Node(new_state);
        // std::cout << "in for *3" << std::endl;
        std::string key = boost::python::extract<std::string>(move.attr("uci")());
        // std::cout << "in for *4" << std::endl;
        double val = actions[key];
		uint64_t edge = leaf->add_child(child, move, val);
	}

    // std::cout << "returning expand" << std::endl;
	return (uint64_t)leaf;
}

uint64_t C_MCTS::backpropagate(C_Node* end_node, boost::multiprecision::cpp_dec_float_50 value){
	for(int i=0; i<this->game_path.size(); ++i){
		this->game_path[i]->input_node->N += 1;
		this->game_path[i]->N += 1;
		this->game_path[i]->W += value;
	}

	return (uint64_t)end_node;
}

bool C_MCTS::move_root(uint64_t move0, uint64_t move1){
    boost::python::object action0 = ((C_Edge*)move0)->action;
    boost::python::object action1 = ((C_Edge*)move1)->action;
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
            root = e->edges[i]->output_node;
            return true;
        }
    }

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

boost::python::object C_MCTS::get_all_edges(boost::python::object lst){
    for(int i=0; i < root->edges.size(); ++i){
        lst.attr("append")((uint64_t)root->edges[i]);
    }

    return lst;
}

boost::python::object C_MCTS::get_edge_action(uint64_t edge){
    return ((C_Edge*)edge)->action;
}

std::string C_MCTS::get_edge_uci(uint64_t edge){
    return boost::python::extract<std::string>(((C_Edge*)edge)->action.attr("uci")());
}

// class shit{
//     public:
//         int N;

//         int getN(){
//             return N;
//         }

//         shit(){
//             N = 6969;
//         }
// };

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
        .def(boost::python::init<C_Node*, C_Node*, boost::python::object, boost::multiprecision::cpp_dec_float_50>())
        .def("upper_confidence_bound", &C_Edge::upper_confidence_bound)
        ;

    boost::python::class_ <C_MCTS>("MCTS")
        .def(boost::python::init<>())
        .def(boost::python::init<boost::python::object, std::string, bool>())
        .def("run_simulations", &C_MCTS::run_simulations)
        // .def("select_child", &C_MCTS::select_child)
        // .def("map_valid_move", &C_MCTS::map_valid_move)
        // .def("probabilities_to_actions", &C_MCTS::probabilities_to_actions)
        // .def("expand", &C_MCTS::expand)
        // .def("backpropagate", &C_MCTS::backpropagate)
        .def("move_root", &C_MCTS::move_root)
        .def("get_sum_N", &C_MCTS::get_sum_N)
        .def("get_edge_N", &C_MCTS::get_edge_N)
        .def("get_all_edges", &C_MCTS::get_all_edges)
        .def("get_edge_action", &C_MCTS::get_edge_action)
        .def("get_edge_uci", &C_MCTS::get_edge_uci)
        ;

    // boost::python::class_<shit>("shit")
    //     .def(boost::python::init<>())
    //     .def("getN", &shit::getN)
    //     ;
}

// int main(){
//     // std::cout << "hello" << std::endl;
// }
