#include "dead_code_elimination.h"

#include "torch/csrc/jit/passes/alias_analysis.h"

#include <unordered_map>

namespace torch {
namespace jit {

class DeadCodeEliminator {
 public:
  // If given a top-level graph, DCE will construct an aliasDb that allows for
  // "smarter" dead code elimination (we will eliminate mutable ops if we can
  // prove the mutated values are not used).
  //
  // Otherwise, we will not allow DCE to eliminate mutable ops.
  explicit DeadCodeEliminator(std::shared_ptr<Graph> graph)
      : aliasDb_(AliasAnalysis(graph)) {}
  DeadCodeEliminator(){};

  void run(Block* block, bool recurse) {
    auto nodes = block->nodes().reverse();
    for (auto it = nodes.begin(); it != nodes.end(); it++) {
      auto node = *it;
      // note these occur before the recursion because we want to uncover
      // dead code in the blocks used to calculate the output
      removeDeadIfOutputs(node);
      removeDeadLoopOutputs(node);
      if (recurse) {
        for (Block* block : node->blocks())
          run(block, true);
      }
      if (!node->hasUses() && !hasSideEffects(node))
        it.destroyCurrent();
    }
  }

 private:
  // If true, disallow DCE because this node mutates a value that is used later
  bool hasLiveWrites(Node* node) {
    // onnx export calls EliminateDeadCode but sometimes passes invalid
    // aten operators. So we call maybeSchema so we handle the cases when
    // there is no valid schema for a node
    const auto schema = node->maybeSchema();
    if (!node->kind().is_aten() || !schema) {
      return false;
    }

    if (!aliasDb_) {
      // If we don't have alias information, disallow all DCE for mutable ops
      return schema->is_mutable();
    } else {
      return aliasDb_->hasLiveWrites(node);
    }
  }

  bool hasSideEffects(Node* node) {
    // FIXME: PythonOp should be treated as having side effects as well!
    //        Unfortunately ONNX depends on it getting removed in this pass, so
    //        it's not a simple change.
    auto it = memo_.find(node);
    if (it != memo_.end())
      return it->second;
    bool has_side_effects = node->kind() == prim::Print ||
        node->kind() == prim::RaiseException ||
        std::any_of(node->blocks().begin(),
                    node->blocks().end(),
                    [&](Block* b) {
                      return std::any_of(
                          b->nodes().begin(), b->nodes().end(), [&](Node* n) {
                            return hasSideEffects(n);
                          });
                    }) ||
        hasLiveWrites(node);

    memo_.emplace(node, has_side_effects);
    return has_side_effects;
  }

  void removeDeadIfOutputs(Node* node) {
    if (node->kind() != prim::If)
      return;
    for (size_t i_1 = node->outputs().size(); i_1 > 0; --i_1) {
      size_t i = i_1 - 1;
      if (!node->outputs().at(i)->hasUses()) {
        node->eraseOutput(i);
        for (Block* b : node->blocks()) {
          b->eraseOutput(i);
        }
      }
    }
  }

  void removeDeadLoopOutputs(Node* node) {
    if (node->kind() != prim::Loop)
      return;
    auto loop_body = node->blocks().at(0);
    auto loop_input_offset = 2; // offset of loop carried deps in input list
    auto loop_body_offset =
        1; // offset to the loop carried dependencies in block inputs/outputs

    for (size_t i_1 = node->outputs().size(); i_1 > 0; --i_1) {
      size_t i = i_1 - 1;
      if (!node->outputs().at(i)->hasUses() &&
          !loop_body->inputs().at(loop_body_offset + i)->hasUses()) {
        node->eraseOutput(i);
        node->removeInput(loop_input_offset + i);
        loop_body->eraseInput(loop_body_offset + i);
        loop_body->eraseOutput(loop_body_offset + i);
      }
    }
  }

  c10::optional<AliasDb> aliasDb_;
  std::unordered_map<Node*, bool> memo_;
};

void EliminateDeadCode(const std::shared_ptr<Graph>& graph) {
  DeadCodeEliminator(graph).run(graph->block(), true);
}

void EliminateDeadCode(Block* block, bool recurse) {
  DeadCodeEliminator().run(block, recurse);
}

} // namespace jit
} // namespace torch
