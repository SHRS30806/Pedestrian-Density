import re

with open('inference_demo.py', 'r', encoding='utf-8') as f:
    text = f.read()

# Fix the broken print statements
text = text.replace('    print(\"\n🚦 Starting inference episode...\"    print(f\"Initial phase: {current_phase.name}\")', '    print(\"\\n🚦 Starting inference episode...\")\n    print(f\"Initial phase: {current_phase.name}\")')

text = text.replace('    print(\"\n🏁 Episode completed!\"    print(f\"Total steps: {len(episode_data[\'actions\'])}\")', '    print(\"\\n🏁 Episode completed!\")\n    print(f\"Total steps: {len(episode_data[\'actions\'])}\")')

text = text.replace('    print(\"\n📊 Decision Analysis:\"    print(f\"  - Total actions taken: {len(actions)}\")', '    print(\"\\n📊 Decision Analysis:\")\n    print(f\"  - Total actions taken: {len(actions)}\")')

text = text.replace('    print(\"\n✅ Inference demo completed!\"    print(f\"💡 The trained PPO agent successfully controlled traffic signals,\")', '    print(\"\\n✅ Inference demo completed!\")\n    print(f\"💡 The trained PPO agent successfully controlled traffic signals,\")')

with open('inference_demo.py', 'w', encoding='utf-8') as f:
    f.write(text)
