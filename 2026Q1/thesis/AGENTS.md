# Objective
任务目标是完成一个调度器，借鉴tomasulo算法，来优化vllm在npu芯片上的性能。

# Baseline
该项目的 baseline 是 vllm 的原生调度器，使用 async scheduler, model_runner_v2 这两个功能。

# algorithm

## 一些数据结构

Reservation Station (RS)

- 将需要组成batch的entry分类为三种类型的entry，用三个不同的列表来表示，
    1. ready_for_forwarding 这种条目可以集成到 input_batch 中，所有的条件都已经准备就绪，可以将该条目分为Decode阶段, prefill阶段，默认支持chunked prefill，让Decode阶段的entry优先进入input_batch，然后再考虑prefill阶段。假设该ready_for_forwarding组成的input_batch_i被全部发射完成之后立马通知scheduler，使得scheduler可以将prescheduled_entry送入RS，那么continuous_forming_input_batch_threading就可以处理这些entry。
    2. prescheduled_entry 这种条目是预发射的，比如发射了input_batch_i之后，可以立即通过prescheduled_entry 来形成prescheduled_input_batch_i+1。这个prescheduled_input_batch_i+1需要等待CDB解决token dependency后（即在对应位置上填充好对应的token）可以直接成为input_batch_i+1，并将此input_batch_i+1送入function_execute_threading后立即去通过prescheduled entry来形成prescheduled_input_batch_i+2，实现input_batch形成和model forward并行执行。
    3. waiting entry 这种条目虽然已经被调度，但是仍有一些没有条件没有满足，不可被集成到input_batch，但其运行的显存(page_kv_cache)等资源已经被分配成功，转化为ready_for_forwarding很快（比如在PD分离架构或者cpu_kv_offloading等架构，其page_kv_cache分配已经完毕，但是还没传输拷贝到对应的page位置）。注：在实验demo中，为了保持简洁性，不会实现这种entry。

Common Data Bus (CDB)

- 该结构负责广播一些消息给各个线程，RS，或者事件，让那些线程能够处理一些事情。
- 广播的消息有：token dependency (解决token依赖问题)，request finished (解决显存释放问题)
- 在实现上，我们不用广播，而是用event来通知各个线程。

## 一些线程或进程

schedule_threading(Scheduler)：

- 该线程将进行调度，把到来的request分配好kv_cache_block的资源（采用page attention策略），并转化为对应的entry存入RS中，如果没有充足的资源，则将其放入waitting队列，并在有充足资源的情况下，将waitting队列的队首取出，分配资源。
- 该scheduler会在`continuous_forming_input_batch_threading`发射input_batch_i后去立即形成preschedule_entry给RS，然后`continuous_forming_input_batch_threading`收到这些entry之后就可以形成prescheduled_input_batch_i+1，然后等待token_dependency解决后变成input_batch。
- 同时，接受CDB发过来的request finished消息，将对应的kv_cache_page标记上free，可以分配给其他request。

continuous_forming_input_batch_threading(Assembler)：该线程使用**事件驱动(Event-driven)**机制，只要发现了ready状态的条目，就可以持续集成input_batch。

- 设立一个check机制，如果处于model_forward_busy状态下，可以持续集成input_batch。如果处于model_forward_idle状态下，但是scheduler中有尚未形成entry的请求，有机会等几毫秒（time_out），通过凑更多input_batch以提高GPU利用率，则可继续集成，而如果超过该设定好的time_out，则强制发射该input_batch。如果处于model_forward_idle状态，且无机会提高GPU利用率，则直接将形成的input_batch送入model_forwarding_thread。
- 默认采用chunked prefill策略，设置一个token_budget（如2048），先填充decode token，再填充prefill token。
- 值得注意的是，当发射完input_batch_i后，还需要找到一个方法，将preschedule_input_batch_i+1和新接收到的需要prefill的ready_for_forwarding entry结合在一起，共同形成一个preschedule_input_batch_i+1。

function_execute_threading(Executor)：该线程负责执行function（最开始将其设计为整个的model_forwarding函数）

- 如果没收到input_batch，则等待，一旦受到则立刻执行forward。
- forward结束之后通过CDB将生成的token发送到prescheduled_batch的待填充槽位。
- 发送完CDB之后继续等待input_batch到来。

## 我的目标：

为了验证该方案的可行性，我需要尽可能快地做出实验验证，证明我的方案可以带来性能的提升，减少CPU-GPU的串行等待时间。所以我准备做一个初步的调度器设计，目的是做一个尽可能简单，实现快捷的实验性demo，不要过多的考虑鲁棒性和corner case。

## 暂不考虑的内容：

当显存不够需要抢占和恢复的情况：为了使实现保持快速和简洁，该情况暂不考虑。

## 推理过程

当一个请求到达后：

1. Scheduler: Do schedule, i.e. Allocate `kv cache` and then add one entry to RS.
2. Batch Assembler: Extract entry to forming `Batch_i`
3. Batch Assembler: Dispatch `Batch_i` to Executor, Meanwhile inform the Scheduler to start preschedule.
4. 4.1 Executor: Do LLM forwarding. **Meanwhile** 4.2 Scheduler do preschedule, and then 4.3 Batch Assembler: forming the `prescheduled_Batch_i+1`. In this way, we can overlap the LLM forwarding and assembling Batch. In general, the `prescheduled_Batch_i+1` will finish before LLM forwarding.
5. Executor: Send back generated token to `prescheduled_Batch_i+1` and it become the `Batch_i+1`, then the Batch Assembler Dispatch `Batch_i+1` to Executor. If the generated tokens contain `<eos>`, then the Executor will inform the Scheduler to remove the corresponding request.