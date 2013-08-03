#ifndef GRAPHS_GRAPHSDATATYPES_H
#define GRAPHS_GRAPHSDATATYPES_H

#include<utilities/Debug.h>
#include<utilities/Printer.h>

#include<vector>
#include<iostream>
#include<iterator>
#include<algorithm>
#include<cassert>

namespace graphs {
  typedef int IDType;
  static IDType Counter = 0;
  /**
   * @class Vertex
   * The Vertex data structure keeps track of edges only by
   * their IDs. There should be a mechanism to retrieve the
   * Edges from their respective IDs. It also means that IDs
   * for each edge should be unique.
   * @todo: Include a mechanism to annotate edges
   * w.r.t. direction i.e., incomong/outgoing.
   */
  template<typename EdgeType, typename WeightType>
  class Vertex {
    public:
      typedef typename std::vector<IDType> EdgesType;

      // Construct a single node.
      Vertex(WeightType w)
        : Weight(w), Id(Counter++)
      {
        DEBUG3(dbgs() << "\nConstructing Vertex:" << GetId());
      }

      /*~Vertex()
      {
        DEBUG(dbgs() << "\nDeleting Vertex:" << GetId());
      }*/
      // Construct a vertex when it has edges.
      //Vertex(EdgesType e, WeightType w)
      //  : Edges(e), Weight(w), Id(Counter++)
      //{ }

      void AddEdge(EdgeType* e)
      {
        DEBUG3(dbgs() << "\nAdding edge: " << e->GetId()
                     << "to neuron: " << GetId());
        auto it = std::find(Edges.begin(), Edges.end(), e->GetId());
        if(it == Edges.end())
          Edges.push_back(e->GetId());
        else
          DEBUG3(dbgs() << "\nEdge: " << *it
                << "already in Neuron:" << GetId());
      }

      void RemoveEdge(EdgeType* e)
      {
        DEBUG3(dbgs() << "Removing Edge: " << e->GetId()
                      << "From Vertex: " << GetId());
        auto it = std::find(Edges.begin(), Edges.end(), e->GetId());
        assert(it != Edges.end() &&
               "No Such Edge connected to this vertex");
        Edges.erase(it);
      }

      bool HasEdge(EdgeType* e)
      {
        return std::find(Edges.begin(), Edges.end(), e->GetId())
                != Edges.end();
      }

      void SetWeight(const WeightType& w)
      {
        Weight = w;
      }

      const EdgesType& GetEdges() const
      {
        return Edges;
      }

      const WeightType& GetWeight() const
      {
        return Weight;
      }

      IDType GetId() const
      {
        return Id;
      }

      void print(std::ostream& os,
                 const Vertex<EdgeType, WeightType>& v)
      {
        os << v;
      }

      void dump() const
      {
        print(dbgs(), *this);
      }

      friend
      std::ostream& operator<<(std::ostream& os, const EdgesType& e)
      {
        os << "(";
        for (auto i = e.begin(); i != e.end(); ++i)
          os << *i << ", ";
        os << ")";
        return os;
      }

      friend
      std::ostream& operator<<(std::ostream& os,
                          const Vertex<EdgeType, WeightType>& t)
      {
        if (printer::YAML) {
          os << "\n- ID: " << t.Id
             << "\n\t- Weight: " << t.Weight
             << "\n\t- Edges: ";
          for (auto it = t.Edges.begin(); it != t.Edges.end(); ++it) {
            os << " " << *it;
          }
        } else {
          os << "(W:" << t.Weight << ", E:" << t.Edges() << ")";
        }
        return os;
      }
    private:
      EdgesType Edges;
      WeightType Weight;
      IDType Id;
  };

  /**
   * @class Edge
   * Each edge has a unique ID in the graph.
   * The nodes attached to an edge are denoted by
   * InNode and OutNode. In case of directed graph, this
   * might not make much sense though.
   */ 
  template<typename NodeType, typename WeightType>
  class Edge {
    public:
      Edge(NodeType& In, NodeType& Out, WeightType w)
      : InNode(&In), OutNode(&Out), Weight(w), Id(Counter++)
      {
        DEBUG3(dbgs() << "\nConstructing Edge:" << GetId());
      }

      /*~Edge()
      {
        DEBUG(dbgs() << "\nDeleting Edge:" << GetId());
      }*/
      void SetWeight(const WeightType& w)
      {
        Weight = w;
      }

      const WeightType& GetWeight() const
      {
        return Weight;
      }

      IDType GetId() const
      {
        return Id;
      }

      NodeType* GetInNode()
      {
        return InNode;
      }

      NodeType* GetOutNode()
      {
        return OutNode;
      }

      void print(std::ostream& os,
                 const Edge<NodeType, WeightType>& e)
      {
        os << e;
      }

      void dump() const
      {
        print(dbgs(), *this);
      }

      friend std::ostream& operator<<(std::ostream& os,
                            const Edge<NodeType, WeightType>& e)
      {
        if (printer::YAML) {
        os << "- ID: " << e.GetId()
           << "\n\t- In: " << e.InNode->GetId()
           << "\n\t- Out: " << e.OutNode->GetId()
           << "\n\t- Weight: " << e.Weight;
        } else {
          os << "(W:" << e.Weight << ", E:" << e.GetId() << ")";
        }
        return os;
      }

    protected:
      NodeType* InNode;
      NodeType* OutNode;
    private:
      WeightType Weight;
      IDType Id;
  };
} // namespace graphs
#endif // GRAPHS_GRAPHSDATATYPES_H
