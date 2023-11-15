# Theory

## Q1
The introduction of "soft prompts" represents a significant advancement in addressing the limitations associated with discrete text prompts in large language models. Traditional discrete prompts, such as predefined textual cues, often suffer from brittleness and lack of flexibility. Soft prompts, on the other hand, allow for a more nuanced and continuous form of conditioning. Rather than relying on fixed and rigid prompts, soft prompts enable the model to consider a distribution of possible prompt variations, providing a smoother and more adaptable mechanism for task-specific conditioning. This approach is particularly beneficial when dealing with diverse tasks or when a task requires a range of input formulations, allowing the model to capture a broader contextual understanding and generate more contextually relevant outputs.

Soft prompts are considered a more flexible and efficient approach for task-specific conditioning due to their ability to capture the inherent uncertainty and variability in natural language. The softness allows for a probabilistic interpretation of prompts, enabling the model to weigh different aspects of the input and dynamically adjust its responses based on the given context. This flexibility is crucial for tasks that involve multifaceted or evolving requirements, as soft prompts allow the model to gracefully handle variations in input without relying on a fixed set of prompt patterns. The adaptability of soft prompts contributes to improved generalization across different tasks, making them a powerful tool for enhancing the performance and versatility of large language models in various natural language processing applications.

In addition to their flexibility, soft prompts offer the advantage of smaller storage requirements compared to traditional discrete prompts. Traditional prompts, being fixed and discrete, often necessitate the storage of numerous predefined templates or cues, leading to increased storage demands. Soft prompts, however, rely on the representation of a probability distribution over potential prompt variations rather than a fixed set of discrete prompts. This probabilistic nature reduces the need for storing a large number of rigid prompts, resulting in more compact model parameters. As a consequence, models leveraging soft prompts can be more lightweight and resource-efficient, making them particularly advantageous in scenarios where storage constraints are a concern, such as deploying models on edge devices or in environments with limited storage capacity. The reduced storage requirements of soft prompts contribute to the overall efficiency and accessibility of large language models, facilitating their deployment in a wider range of practical applications.

## Q2
The efficiency of prompt tuning exhibits an interesting relationship with the scale of language models, and this connection holds significant implications for the future development of large-scale language models and their adaptability to specific tasks. As the scale of a language model increases, with larger models having more parameters and capturing more complex patterns in data, prompt tuning becomes a more potent and versatile tool. Larger models inherently possess a higher capacity to understand and respond to a diverse range of prompts, allowing for more nuanced and contextually rich adaptations to specific tasks. The increased efficiency in prompt tuning with larger models is particularly evident when dealing with multifaceted or domain-specific tasks, as these models can leverage their extensive knowledge to generate more accurate and relevant outputs based on subtle nuances in the provided prompts.

However, this relationship also introduces considerations related to computational resources and deployment feasibility. While larger models demonstrate enhanced efficiency in prompt tuning, they come with higher computational requirements, potentially limiting their applicability in resource-constrained environments. Striking a balance between model scale and resource efficiency becomes crucial for future developments. Additionally, advancements in model compression techniques or the exploration of efficient model architectures may play a pivotal role in mitigating the computational challenges associated with large-scale models. Future developments in large-scale language models must carefully navigate these considerations to ensure that the adaptability gained through prompt tuning is accessible across diverse use cases without imposing prohibitive computational burdens.

# params
batch_size = 2
soft_prompt_length = 4

soft prompts used:
- qna --  "question answer help result"
- sum -- "summarize shorten distill short"
- tx -- "bulgarian english translation translate_bg_en"

CRL used, as per the huggingface model loss (standard procedure)
AdamW optimiser used
lr = 5e-5
num_epochs = 10
save_interval = 3

a subset of all datasets has been used to save on time and evaluate quicker

# qna loss
Epoch: 0 | Step: 0 | Loss: 90.364143
Epoch: 0 | Step: 500 | Loss: 92.699394
Epoch: 0 | Step: 1000 | Loss: 81.093201
Epoch: 0 | Step: 1500 | Loss: 81.144226
Epoch: 0 | Step: 2000 | Loss: 61.287262
Epoch: 0 | Step: 2500 | Loss: 66.911339
Epoch: 0 | Step: 3000 | Loss: 59.616497
Epoch: 0 | Step: 3500 | Loss: 42.372711
Epoch: 0 | Step: 4000 | Loss: 48.118244
Epoch: 0 | Step: 4500 | Loss: 29.414433
Epoch: 1 | Average train loss: 60.550224
Epoch: 1 | Average val loss: 261.075073
Epoch: 1 | Step: 0 | Loss: 23.284517
Epoch: 1 | Step: 500 | Loss: 23.517591
Epoch: 1 | Step: 1000 | Loss: 12.355154
Epoch: 1 | Step: 1500 | Loss: 11.793881
Epoch: 1 | Step: 2000 | Loss: 10.715250
Epoch: 1 | Step: 2500 | Loss: 9.522204
Epoch: 1 | Step: 3000 | Loss: 13.145060
Epoch: 1 | Step: 3500 | Loss: 10.203980
Epoch: 1 | Step: 4000 | Loss: 8.492826
Epoch: 1 | Step: 4500 | Loss: 9.738296
Epoch: 2 | Average train loss: 12.339457
Epoch: 2 | Average val loss: 97.505432
Epoch: 2 | Step: 0 | Loss: 10.415687
Epoch: 2 | Step: 500 | Loss: 9.597601
Epoch: 2 | Step: 1000 | Loss: 8.700298
Epoch: 2 | Step: 1500 | Loss: 9.569082
Epoch: 2 | Step: 2000 | Loss: 8.488832
Epoch: 2 | Step: 2500 | Loss: 13.726034
Epoch: 2 | Step: 3000 | Loss: 10.298596
Epoch: 2 | Step: 3500 | Loss: 9.438914
Epoch: 2 | Step: 4000 | Loss: 8.936327
Epoch: 2 | Step: 4500 | Loss: 10.090421
Epoch: 3 | Average train loss: 9.309041
Epoch: 3 | Average val loss: 89.974091
Epoch: 3 | Step: 0 | Loss: 8.259614
Epoch: 3 | Step: 500 | Loss: 7.590735
Epoch: 3 | Step: 1000 | Loss: 8.556646
Epoch: 3 | Step: 1500 | Loss: 10.325963
Epoch: 3 | Step: 2000 | Loss: 9.302973
Epoch: 3 | Step: 2500 | Loss: 8.045160
Epoch: 3 | Step: 3000 | Loss: 7.825453
Epoch: 3 | Step: 3500 | Loss: 9.157065
Epoch: 3 | Step: 4000 | Loss: 7.411917
Epoch: 3 | Step: 4500 | Loss: 8.781672
Epoch: 4 | Average train loss: 8.798200
Epoch: 4 | Average val loss: 85.970772
Epoch: 4 | Step: 0 | Loss: 8.456946
Epoch: 4 | Step: 500 | Loss: 8.990714
Epoch: 4 | Step: 1000 | Loss: 7.619286
Epoch: 4 | Step: 1500 | Loss: 8.105558
Epoch: 4 | Step: 2000 | Loss: 10.572968
Epoch: 4 | Step: 2500 | Loss: 8.009337
Epoch: 4 | Step: 3000 | Loss: 8.347687
Epoch: 4 | Step: 3500 | Loss: 10.374784
Epoch: 4 | Step: 4000 | Loss: 11.096118
Epoch: 4 | Step: 4500 | Loss: 7.877007
Epoch: 5 | Average train loss: 8.445421
Epoch: 5 | Average val loss: 83.096458
Epoch: 5 | Step: 0 | Loss: 7.966483
Epoch: 5 | Step: 500 | Loss: 8.153994
Epoch: 5 | Step: 1000 | Loss: 8.728663
Epoch: 5 | Step: 1500 | Loss: 7.632365
Epoch: 5 | Step: 2000 | Loss: 8.869977
Epoch: 5 | Step: 2500 | Loss: 8.756426
Epoch: 5 | Step: 3000 | Loss: 9.197432
Epoch: 5 | Step: 3500 | Loss: 7.481238
Epoch: 5 | Step: 4000 | Loss: 7.926927
Epoch: 5 | Step: 4500 | Loss: 7.889951
Epoch: 6 | Average train loss: 8.205209
Epoch: 6 | Average val loss: 81.231888
Epoch: 6 | Step: 0 | Loss: 8.563473
Epoch: 6 | Step: 500 | Loss: 6.638690
Epoch: 6 | Step: 1000 | Loss: 7.563390
Epoch: 6 | Step: 1500 | Loss: 7.099288
Epoch: 6 | Step: 2000 | Loss: 7.043757
Epoch: 6 | Step: 2500 | Loss: 7.903889
Epoch: 6 | Step: 3000 | Loss: 7.541026
Epoch: 6 | Step: 3500 | Loss: 7.967153
Epoch: 6 | Step: 4000 | Loss: 6.752001
Epoch: 6 | Step: 4500 | Loss: 7.071021
Epoch: 7 | Average train loss: 8.031654
Epoch: 7 | Average val loss: 79.585892
Epoch: 7 | Step: 0 | Loss: 8.085259
Epoch: 7 | Step: 500 | Loss: 8.391776
Epoch: 7 | Step: 1000 | Loss: 7.710935
Epoch: 7 | Step: 1500 | Loss: 8.253292
Epoch: 7 | Step: 2000 | Loss: 8.757097
Epoch: 7 | Step: 2500 | Loss: 7.179378
Epoch: 7 | Step: 3000 | Loss: 7.708841
Epoch: 7 | Step: 3500 | Loss: 8.428192
Epoch: 7 | Step: 4000 | Loss: 9.293462
Epoch: 7 | Step: 4500 | Loss: 6.873972
Epoch: 8 | Average train loss: 7.911783
Epoch: 8 | Average val loss: 78.754265
Epoch: 8 | Step: 0 | Loss: 7.923544
Epoch: 8 | Step: 500 | Loss: 7.461014
Epoch: 8 | Step: 1000 | Loss: 7.244104
Epoch: 8 | Step: 1500 | Loss: 13.526814
Epoch: 8 | Step: 2000 | Loss: 8.657562
Epoch: 8 | Step: 2500 | Loss: 10.625204
Epoch: 8 | Step: 3000 | Loss: 8.155709
Epoch: 8 | Step: 3500 | Loss: 8.326866
Epoch: 8 | Step: 4000 | Loss: 8.054008
Epoch: 8 | Step: 4500 | Loss: 7.598687
Epoch: 9 | Average train loss: 7.839058
Epoch: 9 | Average val loss: 78.195869
Epoch: 9 | Step: 0 | Loss: 7.898864
Epoch: 9 | Step: 500 | Loss: 8.132818
Epoch: 9 | Step: 1000 | Loss: 7.926680
Epoch: 9 | Step: 1500 | Loss: 6.889722
Epoch: 9 | Step: 2000 | Loss: 7.720600
Epoch: 9 | Step: 2500 | Loss: 8.087747
Epoch: 9 | Step: 3000 | Loss: 7.453268
Epoch: 9 | Step: 3500 | Loss: 8.142410
Epoch: 9 | Step: 4000 | Loss: 7.232144
Epoch: 9 | Step: 4500 | Loss: 11.366303
Epoch: 10 | Average train loss: 7.804620
Epoch: 10 | Average val loss: 77.961517

# sum loss
Epoch: 0 | Step: 0 | Loss: 90.364143
Epoch: 0 | Step: 500 | Loss: 92.699394
Epoch: 0 | Step: 1000 | Loss: 81.093201
Epoch: 0 | Step: 1500 | Loss: 81.144226
Epoch: 0 | Step: 2000 | Loss: 61.287262
Epoch: 0 | Step: 2500 | Loss: 66.911339
Epoch: 0 | Step: 3000 | Loss: 59.616497
Epoch: 0 | Step: 3500 | Loss: 42.372711
Epoch: 0 | Step: 4000 | Loss: 48.118244
Epoch: 0 | Step: 4500 | Loss: 29.414433
Epoch: 1 | Average train loss: 60.550224
Epoch: 1 | Average val loss: 261.075073
Epoch: 1 | Step: 0 | Loss: 23.284517
Epoch: 1 | Step: 500 | Loss: 23.517591
Epoch: 1 | Step: 1000 | Loss: 12.355154
Epoch: 1 | Step: 1500 | Loss: 11.793881
Epoch: 1 | Step: 2000 | Loss: 10.715250
Epoch: 1 | Step: 2500 | Loss: 9.522204
Epoch: 1 | Step: 3000 | Loss: 13.145060
Epoch: 1 | Step: 3500 | Loss: 10.203980
Epoch: 1 | Step: 4000 | Loss: 8.492826
Epoch: 1 | Step: 4500 | Loss: 9.738296
Epoch: 2 | Average train loss: 12.339457
Epoch: 2 | Average val loss: 97.505432
Epoch: 2 | Step: 0 | Loss: 10.415687
Epoch: 2 | Step: 500 | Loss: 9.597601
Epoch: 2 | Step: 1000 | Loss: 8.700298
Epoch: 2 | Step: 1500 | Loss: 9.569082
Epoch: 2 | Step: 2000 | Loss: 8.488832
Epoch: 2 | Step: 2500 | Loss: 13.726034
Epoch: 2 | Step: 3000 | Loss: 10.298596
Epoch: 2 | Step: 3500 | Loss: 9.438914
Epoch: 2 | Step: 4000 | Loss: 8.936327
Epoch: 2 | Step: 4500 | Loss: 10.090421
Epoch: 3 | Average train loss: 9.309041
Epoch: 3 | Average val loss: 89.974091
Epoch: 3 | Step: 0 | Loss: 8.259614
Epoch: 3 | Step: 500 | Loss: 7.590735
Epoch: 3 | Step: 1000 | Loss: 8.556646
Epoch: 3 | Step: 1500 | Loss: 10.325963
Epoch: 3 | Step: 2000 | Loss: 9.302973
Epoch: 3 | Step: 2500 | Loss: 8.045160
Epoch: 3 | Step: 3000 | Loss: 7.825453
Epoch: 3 | Step: 3500 | Loss: 9.157065
Epoch: 3 | Step: 4000 | Loss: 7.411917
Epoch: 3 | Step: 4500 | Loss: 8.781672
Epoch: 4 | Average train loss: 8.798200
Epoch: 4 | Average val loss: 85.970772
Epoch: 4 | Step: 0 | Loss: 8.456946
Epoch: 4 | Step: 500 | Loss: 8.990714
Epoch: 4 | Step: 1000 | Loss: 7.619286
Epoch: 4 | Step: 1500 | Loss: 8.105558
Epoch: 4 | Step: 2000 | Loss: 10.572968
Epoch: 4 | Step: 2500 | Loss: 8.009337
Epoch: 4 | Step: 3000 | Loss: 8.347687
Epoch: 4 | Step: 3500 | Loss: 10.374784
Epoch: 4 | Step: 4000 | Loss: 11.096118
Epoch: 4 | Step: 4500 | Loss: 7.877007
Epoch: 5 | Average train loss: 8.445421
Epoch: 5 | Average val loss: 83.096458
Epoch: 5 | Step: 0 | Loss: 7.966483
Epoch: 5 | Step: 500 | Loss: 8.153994
Epoch: 5 | Step: 1000 | Loss: 8.728663
Epoch: 5 | Step: 1500 | Loss: 7.632365
Epoch: 5 | Step: 2000 | Loss: 8.869977
Epoch: 5 | Step: 2500 | Loss: 8.756426
Epoch: 5 | Step: 3000 | Loss: 9.197432
Epoch: 5 | Step: 3500 | Loss: 7.481238
Epoch: 5 | Step: 4000 | Loss: 7.926927
Epoch: 5 | Step: 4500 | Loss: 7.889951
Epoch: 6 | Average train loss: 8.205209
Epoch: 6 | Average val loss: 81.231888
Epoch: 6 | Step: 0 | Loss: 8.563473
Epoch: 6 | Step: 500 | Loss: 6.638690
Epoch: 6 | Step: 1000 | Loss: 7.563390
Epoch: 6 | Step: 1500 | Loss: 7.099288
Epoch: 6 | Step: 2000 | Loss: 7.043757
Epoch: 6 | Step: 2500 | Loss: 7.903889
Epoch: 6 | Step: 3000 | Loss: 7.541026
Epoch: 6 | Step: 3500 | Loss: 7.967153
Epoch: 6 | Step: 4000 | Loss: 6.752001
Epoch: 6 | Step: 4500 | Loss: 7.071021
Epoch: 7 | Average train loss: 8.031654
Epoch: 7 | Average val loss: 79.585892
Epoch: 7 | Step: 0 | Loss: 8.085259
Epoch: 7 | Step: 500 | Loss: 8.391776
Epoch: 7 | Step: 1000 | Loss: 7.710935
Epoch: 7 | Step: 1500 | Loss: 8.253292
Epoch: 7 | Step: 2000 | Loss: 8.757097
Epoch: 7 | Step: 2500 | Loss: 7.179378
Epoch: 7 | Step: 3000 | Loss: 7.708841
Epoch: 7 | Step: 3500 | Loss: 8.428192
Epoch: 7 | Step: 4000 | Loss: 9.293462
Epoch: 7 | Step: 4500 | Loss: 6.873972

# tx loss
Epoch: 0 | Step: 0 | Loss: 90.364143
Epoch: 0 | Step: 500 | Loss: 92.699394
Epoch: 0 | Step: 1000 | Loss: 81.093201
Epoch: 0 | Step: 1500 | Loss: 81.144226
Epoch: 0 | Step: 2000 | Loss: 61.287262
Epoch: 0 | Step: 2500 | Loss: 66.911339
Epoch: 0 | Step: 3000 | Loss: 59.616497
Epoch: 0 | Step: 3500 | Loss: 42.372711
Epoch: 0 | Step: 4000 | Loss: 48.118244
Epoch: 0 | Step: 4500 | Loss: 29.414433
Epoch: 1 | Average train loss: 60.550224
Epoch: 1 | Average val loss: 261.075073
Epoch: 1 | Step: 0 | Loss: 23.284517
Epoch: 1 | Step: 500 | Loss: 23.517591
Epoch: 1 | Step: 1000 | Loss: 12.355154
Epoch: 1 | Step: 1500 | Loss: 11.793881
Epoch: 1 | Step: 2000 | Loss: 10.715250
Epoch: 1 | Step: 2500 | Loss: 9.522204
Epoch: 1 | Step: 3000 | Loss: 13.145060
Epoch: 1 | Step: 3500 | Loss: 10.203980
Epoch: 1 | Step: 4000 | Loss: 8.492826
Epoch: 1 | Step: 4500 | Loss: 9.738296
Epoch: 2 | Average train loss: 12.339457
Epoch: 2 | Average val loss: 97.505432
Epoch: 2 | Step: 0 | Loss: 10.415687
Epoch: 2 | Step: 500 | Loss: 9.597601
Epoch: 2 | Step: 1000 | Loss: 8.700298
Epoch: 2 | Step: 1500 | Loss: 9.569082
Epoch: 2 | Step: 2000 | Loss: 8.488832
Epoch: 2 | Step: 2500 | Loss: 13.726034
Epoch: 2 | Step: 3000 | Loss: 10.298596
Epoch: 2 | Step: 3500 | Loss: 9.438914
Epoch: 2 | Step: 4000 | Loss: 8.936327
Epoch: 2 | Step: 4500 | Loss: 10.090421
Epoch: 3 | Average train loss: 9.309041
Epoch: 3 | Average val loss: 89.974091
Epoch: 3 | Step: 0 | Loss: 8.259614
Epoch: 3 | Step: 500 | Loss: 7.590735
Epoch: 3 | Step: 1000 | Loss: 8.556646
Epoch: 3 | Step: 1500 | Loss: 10.325963
Epoch: 3 | Step: 2000 | Loss: 9.302973
Epoch: 3 | Step: 2500 | Loss: 8.045160
Epoch: 3 | Step: 3000 | Loss: 7.825453
Epoch: 3 | Step: 3500 | Loss: 9.157065
Epoch: 3 | Step: 4000 | Loss: 7.411917
Epoch: 3 | Step: 4500 | Loss: 8.781672
Epoch: 4 | Average train loss: 8.798200
Epoch: 4 | Average val loss: 85.970772
Epoch: 4 | Step: 0 | Loss: 8.456946
Epoch: 4 | Step: 500 | Loss: 8.990714
Epoch: 4 | Step: 1000 | Loss: 7.619286
Epoch: 4 | Step: 1500 | Loss: 8.105558
Epoch: 4 | Step: 2000 | Loss: 10.572968
Epoch: 4 | Step: 2500 | Loss: 8.009337
Epoch: 4 | Step: 3000 | Loss: 8.347687
Epoch: 4 | Step: 3500 | Loss: 10.374784
Epoch: 4 | Step: 4000 | Loss: 11.096118
Epoch: 4 | Step: 4500 | Loss: 7.877007
Epoch: 5 | Average train loss: 8.445421
Epoch: 5 | Average val loss: 83.096458
Epoch: 5 | Step: 0 | Loss: 7.966483
Epoch: 5 | Step: 500 | Loss: 8.153994
Epoch: 5 | Step: 1000 | Loss: 8.728663
Epoch: 5 | Step: 1500 | Loss: 7.632365
Epoch: 5 | Step: 2000 | Loss: 8.869977
Epoch: 5 | Step: 2500 | Loss: 8.756426
Epoch: 5 | Step: 3000 | Loss: 9.197432
Epoch: 5 | Step: 3500 | Loss: 7.481238
Epoch: 5 | Step: 4000 | Loss: 7.926927
Epoch: 5 | Step: 4500 | Loss: 7.889951
Epoch: 6 | Average train loss: 8.205209
Epoch: 6 | Average val loss: 81.231888
Epoch: 6 | Step: 0 | Loss: 8.563473
Epoch: 6 | Step: 500 | Loss: 6.638690
Epoch: 6 | Step: 1000 | Loss: 7.563390
Epoch: 6 | Step: 1500 | Loss: 7.099288
Epoch: 6 | Step: 2000 | Loss: 7.043757
Epoch: 6 | Step: 2500 | Loss: 7.903889
Epoch: 6 | Step: 3000 | Loss: 7.541026
Epoch: 6 | Step: 3500 | Loss: 7.967153
Epoch: 6 | Step: 4000 | Loss: 6.752001
Epoch: 6 | Step: 4500 | Loss: 7.071021