#include <boost/python.hpp>
#include <Python.h>
#include <boost/multiprecision/cpp_dec_float.hpp>
#include <boost/multiprecision/cpp_int.hpp>
#include <cstdlib>
#include <vector>
#include <string>
#include <functional>

//TODO: 1. implement change node function
//TODO: 2. implement functionalities for getting edge action ucis and N values
//TODO: 3. Implement remaining functions

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

        C_Node();
        C_Node(std::string state);
        std::string step(boost::python::object action);
        bool is_game_over();
        bool is_leaf();
        uint64_t add_child(C_Node* child, boost::python::object action, boost::multiprecision::cpp_dec_float_50 prior);
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

        boost::multiprecision::cpp_int N;
        boost::multiprecision::cpp_dec_float_50 W;
        boost::multiprecision::cpp_dec_float_50 P;

        C_Edge();
        C_Edge(C_Node* input_node, C_Node* output_node, boost::python::object action, boost::multiprecision::cpp_dec_float_50 prior);
		boost::multiprecision::cpp_dec_float_50 upper_confidence_bound(boost::multiprecision::cpp_dec_float_50 noise);
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
        std::unordered_map<boost::python::object, boost::multiprecision::cpp_dec_float_50> probabilities_to_actions(std::vector<std::vector<std::vector<boost::multiprecision::cpp_dec_float_50>>> probabilities, std::string board); 
        uint64_t expand(C_Node* leaf);
        uint64_t backpropagate(C_Node* end_node, boost::multiprecision::cpp_dec_float_50 value);

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
    boost::python::object chess_module = boost::python::import("chess");
    // boost::python::object Move_from_chess = chess_module.attr("Move");
    boost::python::object board = chess_module.attr("Board")(this->state);
    // boost::python::object board = Board_function(this->state);
    boost::python::object new_board = board.attr("push")(action);
    std::string new_state = boost::python::extract<std::string>(new_board.attr("fen")());
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

uint64_t C_Node::add_child(C_Node* child, boost::python::object action, boost::multiprecision::cpp_dec_float_50 prior){
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

C_MCTS::C_MCTS(){}

C_MCTS::C_MCTS(boost::python::object agent, std::string state, bool stochastic){
    this->root = new C_Node(state);
    this->agent = agent;
    this->stochastic = stochastic;
}

void C_MCTS::run_simulations(int n){}

uint64_t C_MCTS::select_child(C_Node* node){
    while(! node->is_leaf()){
        if(! node->edges.size()){
            return (uint64_t)node;
        }

        std::vector<boost::multiprecision::cpp_dec_float_50> noise (node->edges.size(), 1);

        if(this->stochastic && node->state == this->root->state){
            // TODO : Put dirichlet noise thing
        }

        C_Edge* best_edge = NULL;
        boost::multiprecision::cpp_dec_float_50 best_score = -1e10;

        for(int i=0; i<node->edges.size(); i++){
            C_Edge* edge = node->edges[i];
            boost::multiprecision::cpp_dec_float_50 score = edge->upper_confidence_bound(noise[i]);
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

    return 0;
}

void C_MCTS::map_valid_move(boost::python::object move){
	boost::python::object from_square = move.attr("from_square");
	boost::python::object to_square = move.attr("to_square");
	boost::python::object chess = boost::python::import("chess");
	boost::python::object plane_index;
	boost::python::object piece = this->cur_board.attr("piece_at")(from_square);
	boost::python::object mapper = boost::python::import("mapper");
	boost::python::object Mapping = mapper.attr("Mapping");

	if(move.attr("promotion") && move.attr("promotion") != chess.attr("QUEEN")){
				boost::python::object x = Mapping.attr("get_underpromotion_move")(move.attr("promotion"), from_square, to_square);
		
		plane_index = Mapping.attr("mapper")[x[0]][1-x[1]];
	}
	else{
		if(piece.attr("piece_type") == chess.attr("KNIGHT")){
			boost::python::object direction = Mapping.attr("get_knight_move")(from_square, to_square);
			plane_index = Mapping.attr("mapper")[direction];
		}
		else{
			boost::python::object x = Mapping.attr("get_queenline_move")(from_square, to_square);
			boost::python::object np = boost::python::import("numpy");
			plane_index = Mapping.attr("mapper")[x[0]][np.attr("abs")(x[1])-1];
		}
	}
	
	boost::python::object row = from_square % 8;
	boost::python::object col = 7 - (from_square / 8);
	// TODO: Ensure correctness here
	this->outputs.push_back({move, plane_index, row, col});
}

// TODO : Need suggestions of how to implement probabilities to actions

uint64_t C_MCTS::expand(C_Node* leaf){
	// boost::python::object chess = boost::python::import("chess");
	// boost::python::object board = chess.attr("Board")(leaf->state);

	// boost::python::object possible_actions = boost::python::extract<list>(board.attr("generate_legal_moves")());

	return 0;
}

uint64_t C_MCTS::backpropagate(C_Node* end_node, boost::multiprecision::cpp_dec_float_50 value){
	for(int i=0; i<this->game_path.size(); ++i){
		this->game_path[i]->input_node->N += 1;
		this->game_path[i]->N += 1;
		this->game_path[i]->W += value;
	}

	return (uint64_t)end_node;
}

class shit{
    public:
        int N;

        int getN(){
            return N;
        }

        shit(){
            N = 6969;
        }
};

BOOST_PYTHON_MODULE(CPP_backend)
{
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
        ;

    boost::python::class_<shit>("shit")
        .def(boost::python::init<>())
        .def("getN", &shit::getN)
        ;
}

// int main(){
//     std::cout << "hello" << std::endl;
// }
