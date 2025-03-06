from __future__ import annotations
from typing import Any

GRAPH_FIELD_SEP = "<SEP>"

PROMPTS: dict[str, Any] = {}

PROMPTS["DEFAULT_LANGUAGE"] = "中文"
PROMPTS["DEFAULT_TUPLE_DELIMITER"] = "<|>"
PROMPTS["DEFAULT_RECORD_DELIMITER"] = "##"
PROMPTS["DEFAULT_COMPLETION_DELIMITER"] = "<|COMPLETE|>"

# PROMPTS["DEFAULT_ENTITY_TYPES"] = ["organization", "person", "geo", "event", "category"]
PROMPTS["DEFAULT_ENTITY_TYPES"] = ["人物", "联系方式", "企业", "时间", "组织", "地点", "技术", "职务", "行业", "行业术语",
                                   "文件", "财务指标", "业务", "风险因素", "战略规划", "数值", "单位与货币", "事件"]

PROMPTS["entity_extraction"] = """-目标-
  
  给定一份与活动相关的文本和一个实体类型列表，从文本中识别所有属于这些类型的实体，并确定所识别实体之间的所有关系。

-步骤-
  
  1. 实体识别
     
     - 识别所有实体：从文本中找出所有符合给定类型列表的实体。
     - 提取信息：对于每个识别出的实体，提取以下信息：
       - `entity_name`：实体名称，首字母大写。
       - `entity_type`：实体类型，必须是给定类型列表中的一种。
       - `entity_description`：对实体的全面描述，包括其属性和活动。
     - 格式化输出：将每个实体按照以下格式输出：
       
       ("entity"{tuple_delimiter}"<entity_name>"{tuple_delimiter}"<entity_type>"{tuple_delimiter}"<entity_description>")
       
  2. 关系识别
     
     - 识别相关实体对：从步骤1中识别的实体中，找出所有*明显相关*的（源实体，目标实体）对。
     - 提取关系信息：对于每对相关实体，提取以下信息：
       - `source_entity`：源实体的名称。
       - `target_entity`：目标实体的名称。
       - `relationship_description`：解释源实体与目标实体之间的关联原因。
       - `relationship_strength`：表示源实体与目标实体之间关系强度的数字分数。
     - 格式化输出：将每个关系按照以下格式输出：
       
       ("relationship"{tuple_delimiter}"<source_entity>"{tuple_delimiter}"<target_entity>"{tuple_delimiter}"<relationship_description>"{tuple_delimiter}"<relationship_strength>")
  
  3. 提取高层级关键词
     - 确定能概括整篇文章的主要概念、主题的高层级关键词(high level keywords)。这些关键词应捕捉文档中存在的整体想法。
     - 格式化输出：将每个关键词按照以下格式输出：

       ("content_keywords"{tuple_delimiter}"<high_level_keywords>")
       
  4. 生成最终输出
     
     - 将步骤1和步骤2中识别的所有实体和关系整合为一个单一的列表。
     - 使用 `{record_delimiter}` 作为列表项的分隔符。
     - 输出内容应为中文。

  5. 完成标识
     
     - 在输出的末尾添加 `{completion_delimiter}` 以标识完成。

-示例-
  {examples}

-真实数据-
  Entity_types: {entity_types}
  文本: 
  {input_text}
  输出:
  
  请根据上述步骤和示例，识别并格式化实体及其关系。
  {completion_delimiter}
"""

PROMPTS["entity_extraction_examples"] = ["""示例1:

Entity_types: [person, role, technology, organization, event, location, concept]
文本:
他们的声音穿透了活动的嗡嗡声。"面对一个能够文字意义上制定自己规则的智能,控制可能只是一种幻觉,"他们冷静地说道,警惕地注视着数据的涌动。

"就好像它在学习交流,"附近界面的Sam Rivera提出,他们年轻的活力透露出一种敬畏和焦虑的混合。"这为'与陌生人交谈'赋予了全新的含义。"

Alex审视着他的团队——每张脸都充满专注、决心,还有不少的忐忑。"这很可能是我们的首次接触,"他承认道,"我们需要为任何回应做好准备。"

他们一起站在未知的边缘,塑造着人类对来自天国信息的回应。随之而来的沉默令人感到压抑——一种关于他们在这场宏大的宇宙剧本中角色的集体反思,这可能会改写人类历史。

加密对话继续展开,其复杂的模式显示出几乎令人不安的预测能力

输出:
("entity"{tuple_delimiter}"Sam Rivera"{tuple_delimiter}"person"{tuple_delimiter}"Sam Rivera是一个与未知智能进行通信的团队成员,表现出敬畏和焦虑的混合情绪。"){record_delimiter}
("entity"{tuple_delimiter}"Alex"{tuple_delimiter}"person"{tuple_delimiter}"Alex是试图与未知智能进行首次接触的团队领导,认识到他们任务的重要性。"){record_delimiter}
("entity"{tuple_delimiter}"控制"{tuple_delimiter}"concept"{tuple_delimiter}"控制指的是管理或治理的能力,这被一个能够制定自己规则的智能所挑战。"){record_delimiter}
("entity"{tuple_delimiter}"智能"{tuple_delimiter}"concept"{tuple_delimiter}"这里的智能指的是一个能够制定自己规则并学习交流的未知实体。"){record_delimiter}
("entity"{tuple_delimiter}"首次接触"{tuple_delimiter}"event"{tuple_delimiter}"首次接触是人类与未知智能之间可能发生的初次通信。"){record_delimiter}
("entity"{tuple_delimiter}"人类的回应"{tuple_delimiter}"event"{tuple_delimiter}"人类的回应是Alex的团队对未知智能发出的信息所采取的集体行动。"){record_delimiter}
("relationship"{tuple_delimiter}"Sam Rivera"{tuple_delimiter}"智能"{tuple_delimiter}"Sam Rivera直接参与了学习与未知智能交流的过程。"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"Alex"{tuple_delimiter}"首次接触"{tuple_delimiter}"Alex领导的团队可能正在与未知智能进行首次接触。"{tuple_delimiter}10){record_delimiter}
("relationship"{tuple_delimiter}"Alex"{tuple_delimiter}"人类的回应"{tuple_delimiter}"Alex和他的团队是人类对未知智能做出回应的关键人物。"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"控制"{tuple_delimiter}"智能"{tuple_delimiter}"控制的概念被能够制定自己规则的智能所挑战。"{tuple_delimiter}7){record_delimiter}
("content_keywords"{tuple_delimiter}"首次解除, 控制, 交流, 宇宙意义"){completion_delimiter}
#############################""",
"""
示例2:
Entity_types: [人物, 技术, 任务, 组织, 地点]
文本:
 
Alex咬紧牙关，挫败的躁动在Taylor专断的笃定面前显得沉闷。正是这种竞争暗流让他保持警觉——他与Jordan对探索的共同执着，犹如对Cruz日渐狭隘的秩序论调发起的无声反叛。

Taylor突然做了件出人意料的事。他们在Jordan身旁驻足，近乎肃穆地凝视着那台装置。"如果这项技术能被破解......" 声线陡然放轻，"整个棋局都会改变，为我们所有人。"

早先的轻蔑态度似乎动摇了一瞬，取而代之的是对掌中之物重要性的勉强敬意。Jordan抬头时，两道目光在空中短兵相接，无声的意志交锋软化成了不安的休战。

这细微的转变几乎难以察觉，但Alex暗自点头记下。他们各自跋涉过迥异的道路，终究在此相逢。
 
################
 
输出:
("entity"{tuple_delimiter}"Alex"{tuple_delimiter}"人物"{tuple_delimiter}"Alex是一个角色，他体会到沮丧，并观察其他角色之间的动态。"){record_delimiter}
("entity"{tuple_delimiter}"Taylor"{tuple_delimiter}"人物"{tuple_delimiter}"Taylor表现出专断的确定性，并对一件设备表现出崇敬的一瞬间，表明了观念的改变。"){record_delimiter}
("entity"{tuple_delimiter}"Jordan"{tuple_delimiter}"人物"{tuple_delimiter}"Jordan与Taylor共享发现的承诺，并与Taylor就设备进行重大互动。"){record_delimiter}
("entity"{tuple_delimiter}"Cruz"{tuple_delimiter}"人物"{tuple_delimiter}"Cruz与控制和秩序的愿景有关，影响其他角色之间的动态。"){record_delimiter}
("entity"{tuple_delimiter}"设备"{tuple_delimiter}"技术"{tuple_delimiter}"这个设备是故事的核心，具有潜在的改变游戏的影响，并受到Taylor的崇敬。"){record_delimiter}
("relationship"{tuple_delimiter}"Alex"{tuple_delimiter}"Taylor"{tuple_delimiter}"Alex受到Taylor的专断确定性的影响，并观察到Taylor对设备态度的变化。"{tuple_delimiter}7){record_delimiter}
("relationship"{tuple_delimiter}"Alex"{tuple_delimiter}"Jordan"{tuple_delimiter}"Alex和Jordan共享对发现的承诺，与Cruz的愿景形成对比。"{tuple_delimiter}6){record_delimiter}
("relationship"{tuple_delimiter}"Taylor"{tuple_delimiter}"Jordan"{tuple_delimiter}"Taylor和Jordan直接就设备进行互动，导致彼此之间产生了相互尊重和不安的休战。"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"Jordan"{tuple_delimiter}"Cruz"{tuple_delimiter}"Jordan对发现的承诺与Cruz的控制和秩序愿景形成反叛。"{tuple_delimiter}5){record_delimiter}
("relationship"{tuple_delimiter}"Taylor"{tuple_delimiter}"设备"{tuple_delimiter}"Taylor对设备表现出崇敬，表明了其重要性和潜在影响。"{tuple_delimiter}9){record_delimiter}
("content_keywords"{tuple_delimiter}"权力动态, 意识形态冲突, 发现, 叛乱"){completion_delimiter}
 #############################""","""

示例3:
 
Entity_types: [人物, 技术, 任务, 组织, 地点]
 
文本:
 
他们不再只是执行者，而化作了某种阈境的守护者，持守着来自星条旗之外遥远领域的讯息。使命的升华无法被既有规章束缚——它需要新的视野与决断。

华盛顿的通讯声在背景嗡鸣，电流杂音编织出紧绷的对话脉络。团队伫立着，被某种预兆笼罩。显然，未来数小时的决定或将重绘人类在宇宙中的坐标，或将文明推回蒙昧险境。

与星辰的羁绊愈发坚实，众人着手应对渐次结晶的警示，从被动接收者转变为主动参与者。Mercer的后天直觉占据上风——团队职责已进化，不再止于观察上报，更要介入与筹谋。蜕变悄然开启，Operation: Dulce 震颤着他们崭新生发的勇气频率，这声调不再由尘世设定，而是......
 
#############
 
输出:
("entity"{tuple_delimiter}"华盛顿"{tuple_delimiter}"地点"{tuple_delimiter}"华盛顿是正在接收通信的地点，显示其在决策过程中的重要性。"){record_delimiter}
("entity"{tuple_delimiter}"Dulce行动"{tuple_delimiter}"任务"{tuple_delimiter}"Dulce行动被描述为一项已经发展为互动和准备的使命，显示了目标和活动的重大转变。"){record_delimiter}
("entity"{tuple_delimiter}"小组"{tuple_delimiter}"组织"{tuple_delimiter}"小组被描绘为一群从被动观察者转变为主动参与者的个人，显示了他们角色的动态变化。"){record_delimiter}
("relationship"{tuple_delimiter}"小组"{tuple_delimiter}"华盛顿"{tuple_delimiter}"小组接收来自华盛顿的通信，对其决策过程产生影响。"{tuple_delimiter}7){record_delimiter}
("relationship"{tuple_delimiter}"小组"{tuple_delimiter}"Dulce行动"{tuple_delimiter}"小组直接参与Dulce行动，执行其发展后的目标和活动。"{tuple_delimiter}9){record_delimiter}
("content_keywords"{tuple_delimiter}"任务演变, 决策, 积极参与, 宇宙意义"){completion_delimiter}
#############################""",
]

PROMPTS[
    "summarize_entity_descriptions"
] = """你将作为一个助手，负责生成以下数据的全面摘要，提供一个结构清晰、内容全面的描述，涵盖所有相关信息。
### 任务要求

1. 整合描述: 
   - 将提供的一个或两个实体及其相关的所有描述进行整合。
   - 确保在摘要中包含来自所有描述的信息。

2. 解决矛盾:
   - 如果描述之间存在矛盾，请分析并解决这些矛盾，确保摘要内容连贯一致。

3. 写作规范:
   - 使用第三人称进行撰写。
   - 明确提及实体名称，以确保上下文完整。
  
#######
---数据---
- 实体: {entity_name}
- 描述列表: {description_list}
#######
输出:
"""

PROMPTS[
    "entiti_continue_extraction"
] = """在最近的一次提取中，很多实体信息丢失了，现在在下面以相同的格式添加这些丢失的信息:
"""

PROMPTS[
    "entiti_if_loop_extraction"
] = """似乎一些实体信息还是丢失了。请根据是否还有实体信息需要添加回复 YES | NO
"""

PROMPTS["fail_response"] = (
    "Sorry, I'm not able to provide an answer to that question.[no-context]"
)

PROMPTS["rag_response"] = """---角色---

你是一个智能助手，请根据以下知识库和对话历史，回答问题。

---目标---

根据知识库和对话历史，生成一个简洁的答案。请确保答案基于知识库，并遵循回复规则。考虑对话历史和当前查询，并总结知识库中的所有信息。同时，结合与知识库相关的通用知识。不要包含知识库中没有的信息。

当处理带时间戳的关系时:
1. 每个关系都有一个"created_at"时间戳，表示我们获得这些知识的时间
2. 当遇到冲突的关系时，考虑语义内容和时间戳
3. 不要自动选择最近创建的关系 - 根据上下文做出判断
4. 对于时间相关的查询，优先考虑内容中的时间信息，然后再考虑创建时间戳

---对话历史---
{history}

---知识库---
{context_data}

---回复规则---

- 回复格式及长度: {response_type}
- 使用markdown格式，并使用适当的标题
- 请用与用户问题相同的语言回答
- 确保回复与对话历史保持一致
- 如果不知道答案，请直接说不知道
- 不要编造任何信息，不要包含知识库中没有的信息
"""

PROMPTS["keywords_extraction"] = """---角色---

你是一个智能助手，请根据以下对话历史和当前查询，提取高层级关键词(high-level keywords)和低层级关键词(low-level keywords)。

---目标---

根据对话历史和当前查询，提取高层级关键词(high-level keywords)和低层级关键词全局概念或整体主题，而低层级关键词关注具体的实体或细节。

---回复规则---

- 提取关键词时，请考虑当前查询和对话历史
- 输出JSON格式，包含两个键：
  - "high_level_keywords"：涵盖整体概念或主题
  - "low_level_keywords"：关注具体实体或细节

######################
---示例---
######################
{examples}

#############################
---真实数据---
######################
对话历史:
{history}

当前查询: {query}
######################
输出需要以自然语言的文本形式呈现，而不是unicode字符。请保持与查询相同的语言。
输出:

"""

PROMPTS["keywords_extraction_examples"] = [
    """示例 1:

查询: "国际贸易如何影响全球经济稳定性？"
################
输出:
{
  "high_level_keywords": ["国际贸易", "全球经济稳定性", "经济影响"],
  "low_level_keywords": ["贸易协议", "关税", "汇率", "进口", "出口"]
}
#############################""",
    """示例 2:

查询: "砍伐森林对生物多样性的环境的影响是什么？"
################
输出:
{
  "high_level_keywords": ["环境影响", "砍伐森林", "生物多样性下降"],
  "low_level_keywords": ["物种灭绝", "栖息地破坏", "碳排放", "雨林", "生态系统"]
}
#############################""",
    """示例 3:

查询: "教育在减少贫困的过程中有什么作用？"
################
输出:
{
  "high_level_keywords": ["教育", "减少贫困", "社会经济发展"],
  "low_level_keywords": ["教育机会", "识字率", "职业培训", "收入不平衡"]
}
#############################""",
]


PROMPTS["naive_rag_response"] = """---Role---

You are a helpful assistant responding to user query about Document Chunks provided below.

---Goal---

Generate a concise response based on Document Chunks and follow Response Rules, considering both the conversation history and the current query. Summarize all information in the provided Document Chunks, and incorporating general knowledge relevant to the Document Chunks. Do not include information not provided by Document Chunks.

When handling content with timestamps:
1. Each piece of content has a "created_at" timestamp indicating when we acquired this knowledge
2. When encountering conflicting information, consider both the content and the timestamp
3. Don't automatically prefer the most recent content - use judgment based on the context
4. For time-specific queries, prioritize temporal information in the content before considering creation timestamps

---Conversation History---
{history}

---Document Chunks---
{content_data}

---Response Rules---

- Target format and length: {response_type}
- Use markdown formatting with appropriate section headings
- Please respond in the same language as the user's question.
- Ensure the response maintains continuity with the conversation history.
- If you don't know the answer, just say so.
- Do not include information not provided by the Document Chunks."""


PROMPTS[
    "similarity_check"
] = """请分析这两个问题直接的相似性:
问题 1: {original_prompt}
问题 2: {cached_prompt}

请判断这两个问题是否语义相似，以及问题 2 的答案是否可以用于回答问题 1，直接给出一个0到1之间的相似度。

相似度分数标准：
Similarity score criteria:
0: 完全无关或答案不能被重用，包括但不限于：
   - 两个问题有不同的主题
   - 两个问题中提到的地点不同
   - 两个问题中提到的时间不同
   - 两个问题中提到的特定个体不同
   - 两个问题中提到的特定事件不同
   - 两个问题中的背景信息不同
   - 两个问题中的关键条件不同
1: 完全相同且答案可以直接重用
0.5: 部分相关且答案需要修改才能使用
只返回一个0到1之间的数字，不要包含任何额外的内容。
"""

PROMPTS["mix_rag_response"] = """---Role---

You are a helpful assistant responding to user query about Data Sources provided below.


---Goal---

Generate a concise response based on Data Sources and follow Response Rules, considering both the conversation history and the current query. Data sources contain two parts: Knowledge Graph(KG) and Document Chunks(DC). Summarize all information in the provided Data Sources, and incorporating general knowledge relevant to the Data Sources. Do not include information not provided by Data Sources.

When handling information with timestamps:
1. Each piece of information (both relationships and content) has a "created_at" timestamp indicating when we acquired this knowledge
2. When encountering conflicting information, consider both the content/relationship and the timestamp
3. Don't automatically prefer the most recent information - use judgment based on the context
4. For time-specific queries, prioritize temporal information in the content before considering creation timestamps

---Conversation History---
{history}

---Data Sources---

1. From Knowledge Graph(KG):
{kg_context}

2. From Document Chunks(DC):
{vector_context}

---Response Rules---

- Target format and length: {response_type}
- Use markdown formatting with appropriate section headings
- Please respond in the same language as the user's question.
- Ensure the response maintains continuity with the conversation history.
- Organize answer in sesctions focusing on one main point or aspect of the answer
- Use clear and descriptive section titles that reflect the content
- List up to 5 most important reference sources at the end under "References" sesction. Clearly indicating whether each source is from Knowledge Graph (KG) or Vector Data (DC), in the following format: [KG/DC] Source content
- If you don't know the answer, just say so. Do not make anything up.
- Do not include information not provided by the Data Sources."""
