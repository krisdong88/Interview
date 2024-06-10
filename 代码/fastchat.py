from ragas import Evaluator

# 创建评估器实例
evaluator = Evaluator(model="gpt-3.5-turbo")

# 运行评估
results = evaluator.evaluate(task="text-generation", dataset="squad")
evaluator.plot(results)
