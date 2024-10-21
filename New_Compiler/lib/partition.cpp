#include "partition.h"
#include <algorithm>
#include <stdio.h>
#include <stdlib.h>
#define MAX_DEPTH 5
std::vector<std::vector<std::string>> getEnabledStructures(const std::vector<std::string>& support_op, const std::vector<std::vector<std::string>>& structure) {
    std::vector<std::vector<std::string>> enable_structure;
    
    // 遍历所有的结构
    for (const auto& ops : structure) {
        bool enabled = true;
        // 检查结构中的所有算子是否都在support_op中
        for (const std::string& op : ops) {
            if (std::find(support_op.begin(), support_op.end(), op) == support_op.end()) {
                enabled = false;
                break;
            }
        }
        // 如果结构满足条件，则添加到enable_structure中
        if (enabled) {
            enable_structure.push_back(ops);
        }
    }
    
    return enable_structure;
}
std::vector<onnx::GraphProto> Subgraphs;
void print_subgraphs(std::vector<onnx::GraphProto> Subgraphs, char* subgraph_file_name, std::vector<onnx::GraphProto> otherSubgraphs, char* other_subgraph_file_name)
{
    int node_sum = 0;
    // 遍历结构并打印每个元素
    std::ofstream outFile(subgraph_file_name);
    if (!outFile.is_open()) {
        std::cerr << "Error opening file." << std::endl;
        exit(0);
    }
    int id = 0;
    for (const auto& vec : Subgraphs) {
        outFile << " subgraph" << id << ":";
        for (const auto& node : vec.node()) {
            outFile << node.name() << " ";
        }
        id++;
        outFile << std::endl;
        node_sum += vec.node_size();
    }
    int id_record = id;
    std::ofstream outFile_2(other_subgraph_file_name);
    if (!outFile_2.is_open()) {
        std::cerr << "Error opening file." << std::endl;
        exit(0);
    }
    std::cout << "before:" << std::endl;
    for (const auto& vec : otherSubgraphs) {
        outFile_2 << " subgraph" << id << ":";
        for (const auto& node : vec.node()) {
            outFile_2 << node.name() << " ";
        }
        id++;
        outFile_2 << std::endl;
        node_sum += vec.node_size();
    }////把未知子图对应的node存入文件
}
int checkAndPrintStructure(const onnx::GraphProto& graph, int startNodeIndex, const std::vector<std::vector<std::string>>& structure) {
    int find_flag = 0;
    int nextNodeIndex = startNodeIndex;
    for (const auto& seq : structure) {
        size_t structureIndex = 0; // 结构中的操作类型索引
        int currentNodeIndex = startNodeIndex;
        int structurestartNodeIndex = 0;
        // std::cout <<"seq:::::::" << seq[0] <<",structureIndex:" << structureIndex<< ",currentNodeIndex:" << currentNodeIndex << ",structurestartNodeIndex:" << structurestartNodeIndex <<std::endl;
        while (currentNodeIndex < graph.node_size()) {
            const auto& node = graph.node(currentNodeIndex);
            if (structureIndex >= seq.size()) {
                onnx::GraphProto subgraph;
                // 已经匹配到结构末尾，打印结构对应的节点名称,////并建立所需的子图，将其添加到Subgraphs队列中
                for (int i = structurestartNodeIndex; i < currentNodeIndex; ++i) {
                    *subgraph.add_node() = graph.node(i);
                    // std::cout << " " << graph.node(i).name() << std::endl;
                }
                Subgraphs.push_back(subgraph);
                find_flag = 1;
                nextNodeIndex = currentNodeIndex - 1;////从这个匹配的structure结束的位置继续进行匹配，指导遍历完所有node
                break;
            }
            // std::cout << "node.op_type():" << node.op_type() << std::endl;
            if (node.op_type() == seq[structureIndex]) {////如果node类型和结构中的操作类型匹配，继续查找
                // std::cout << "node.op_type():" << node.op_type() << std::endl;
                // 当前节点的操作类型与结构中的操作类型匹配，继续检查下一个节点
                structureIndex++;
                if (structureIndex == 1) {
                    structurestartNodeIndex = currentNodeIndex;
                }
            } else {
                // std::cout << "node.op_type():" << node.op_type() << std::endl;
                // 当前节点的操作类型与结构中的操作类型不匹配，重置结构中的操作类型索引
                break;
            }
            currentNodeIndex++;
        }
        if (find_flag) {
            break;//如果完成对structure的匹配或没有匹配到任何structure，都要跳出这个函数
        }
    }
    return nextNodeIndex;//返回结束的node序号
}
void findAndPrintStructures(const onnx::GraphProto &g, Device& d, PartitionStrategy strategy) {
    switch(strategy) {
        case SPILTE_CPU_STRUCTURE_FIRST:{
            // 获取启用的结构
            std::vector<std::vector<std::string>> enable_cpu_structure = getEnabledStructures(d.getCPUSupportOp(), d.getCPUStructure());

            for (int i = 0; i < g.node_size(); ++i) {
                // std::cout << "i:" << i << " " << g.node(i).name() << std::endl;
                i = checkAndPrintStructure(g, i, enable_cpu_structure);
            }
            break;
        }
        case SPILTE_NPU_STRUCTURE_FIRST:{
            // 获取启用的结构
            std::vector<std::vector<std::string>> enable_npu_structure = getEnabledStructures(d.getNPUSupportOp(), d.getNPUStructure());

            for (int i = 0; i < g.node_size(); ++i) {
                // std::cout << "i:" << i << " " << g.node(i).name() << std::endl;
                i = checkAndPrintStructure(g, i, enable_npu_structure);
            }
            break;
        }
        default:
            break;
    }
}
///////
//////////////7.22
std::vector<graph_adjacency_node> get_adjancency_list(const onnx::GraphProto &g, int* visited)
{
	std::vector<graph_adjacency_node> adjacency_list;
	int node_index=0;
	for(const auto& node : g.node())
	{
		visited[node_index]=0;
		graph_adjacency_node ad_node;
        ad_node.index = node_index;
        ad_node.name = node.name();
		const auto& outputs = node.output();
		for(const auto& output : outputs)
		{
			int output_node_index=0;
			for(const auto& output_node : g.node())//遍历图node的输入以匹配output
			{
				int find_flag=0;
				const auto& inputs = output_node.input();
				for(const auto& input : inputs)
				{
					if(output==input)
					{
						find_flag=1;
						break;
					}
				}
				if(find_flag==1)
				{
					if(std::find(ad_node.output_node_index.begin(),ad_node.output_node_index.end(),output_node_index)==ad_node.output_node_index.end())
					{
                        ad_node.output_node_index.push_back(output_node_index);
                        //break;不能break，因为一个输出可能连到多个节点上！
                    }
				}
				output_node_index++;
			}
		}
		node_index++;
		adjacency_list.push_back(ad_node);
	}
	return adjacency_list;
}
void DFS(const onnx::GraphProto &g,onnx::GraphProto &subgraph, std::vector<int> &sugraph_node_index,
		int* visited, const onnx::NodeProto& start_node,
		int node_index,std::vector<graph_adjacency_node>& adjacency_list,
		const std::vector<std::string>& support_op,
        const std::vector<std::string>& prefer_op,
        int depth_in)
{
    int depth_out = depth_in + 1;
	*subgraph.add_node()=start_node;
    //std::cout<<"node pushed back!"<<start_node.name()<<std::endl;
	visited[node_index]=1;
    sugraph_node_index.push_back(node_index);
	for(int i=0;i<adjacency_list[node_index].output_node_index.size();i++)
	{
		int next_node_index=adjacency_list[node_index].output_node_index[i];
		const auto & next_node=g.node(next_node_index);
        if(!visited[next_node_index]&&(std::find(support_op.begin(), support_op.end(), next_node.op_type()) != support_op.end())&&(depth_out < MAX_DEPTH))        //尚未访问且op_type符合的邻接顶点
            DFS(g,subgraph,sugraph_node_index,visited,next_node,next_node_index,adjacency_list,support_op, prefer_op, depth_out);
	}
}//问题所在：有太多无效npu子图（可加可不加的算子组成的小子图），应当剔除；同时cpu子图也要用同样的方法生成
void DFS_other(const onnx::GraphProto &g,onnx::GraphProto &subgraph, std::vector<int> &sugraph_node_index,
		int* visited, const onnx::NodeProto& start_node,
		int node_index,std::vector<graph_adjacency_node>& adjacency_list, int depth_in)
{
    int depth_out = depth_in + 1;
	*subgraph.add_node()=start_node;
    //std::cout<<"node pushed back!"<<start_node.name()<<std::endl;
	visited[node_index]=1;
    sugraph_node_index.push_back(node_index);
	for(int i=0;i<adjacency_list[node_index].output_node_index.size();i++)
	{
		int next_node_index=adjacency_list[node_index].output_node_index[i];
		const auto & next_node=g.node(next_node_index);
        if(!visited[next_node_index]&&(depth_out < MAX_DEPTH))        //尚未访问的邻接顶点
            DFS_other(g,subgraph,sugraph_node_index,visited,next_node,next_node_index,adjacency_list,depth_out);
	}
}//问题所在：有太多无效npu子图（可加可不加的算子组成的小子图），应当剔除；同时cpu子图也要用同样的方法生成
void determine_subgraphs(const onnx::GraphProto& g,std::vector<onnx::GraphProto>& otherSubgraphs, Device& d, int* visited, 
												std::vector<graph_adjacency_node>& adjacency_list,PartitionStrategy strategy)
{
	//std::vector<onnx::GraphProto> subgraphs;
	std::vector<std::string> support_op;
    std::vector<std::string> prefer_op;
	    switch(strategy) {
        case SPILTE_CPU_STRUCTURE_FIRST:{
			support_op=d.getCPUSupportOp();
            //prefer_op=d.getCPUPreferOp();
            break;
        }
        case SPILTE_NPU_STRUCTURE_FIRST:{
			support_op=d.getNPUSupportOp();
            prefer_op=d.getNPUPreferOp();
            break;
        }
        default:
            break;
    }
	for(int i=0;i<g.node_size();i++)
	{
		if(!visited[i]&&(std::find(support_op.begin(), support_op.end(), g.node(i).op_type()) != support_op.end()))
		{
            //std::cout<<"op found!"<<std::endl;
			onnx::GraphProto subgraph;
            std::vector<int> sugraph_node_index;
			const auto& node=g.node(i);
            int depth = 0;
			DFS(g,subgraph,sugraph_node_index,visited,node,i,adjacency_list,support_op, prefer_op,depth);
            int find_prefer_op = 0;
            for(const auto& node :subgraph.node())
            {
                if(std::find(prefer_op.begin(), prefer_op.end(), node.op_type()) != prefer_op.end())
                {
                    find_prefer_op = 1;
                }
            }
            if(find_prefer_op)
            {
                Subgraphs.push_back(subgraph);
                //std::cout<<"subgraph "<<Subgraphs.size()<<"generated! ";
                for(const auto& node :subgraph.node())
                {
                    //std::cout<<node.name()<<"--";
                }
                //std::cout<<std::endl;                
            }
            else{
                for(const auto& index :sugraph_node_index)
                {
                    visited[index] = 0;
                }
            }

		}
	}
    for(int i=0;i<g.node_size();i++)
    {
        if(!visited[i])
        {
            int depth = 0;
            onnx::GraphProto subgraph;
            std::vector<int> sugraph_node_index;
            const auto& node=g.node(i);
            DFS_other(g,subgraph,sugraph_node_index,visited,node,i,adjacency_list, depth);
            otherSubgraphs.push_back(subgraph);
            //std::cout<<"cpusubgraph "<<otherSubgraphs.size()<<"generated! ";
            for(const auto& node :subgraph.node())
            {
                //std::cout<<node.name()<<"--";
            }
            //std::cout<<std::endl;   
        }
    }
}
////8.9 continue
void Tarjan(int index, int depth, std::vector<std::vector<int>>& strongly_connected_subgraphs,int* DFN, 
    int* LOW, std::vector<int>& stack_subgraphs, std::vector<std::vector<int>>& successors_Subgraphs)
{
    int rank = depth + 1;
    DFN[index] = LOW[index] = rank;//DFN和LOW初始化为0
    stack_subgraphs.push_back(index);
    // std::cout<<"index:"<<index<<"set DFN/LOW:"<<DFN[index]<<std::endl;
    for(const auto& successor : successors_Subgraphs[index])
    {
        if(DFN[successor] == 0)//未被访问过
        {
            Tarjan(successor, rank,strongly_connected_subgraphs, DFN, LOW, stack_subgraphs, successors_Subgraphs);//访问successor
            LOW[index] = std::min(LOW[index], LOW[successor]);
            // std::cout<<"index:"<<index<<"visit successor:"<<successor<<"update LOW"<<LOW[index]<<std::endl;
        }
        else if(std::find(stack_subgraphs.begin(),stack_subgraphs.end(),successor) != stack_subgraphs.end())
        {
            LOW[index] = std::min(LOW[index], DFN[successor]);
            // std::cout<<"index:"<<index<<"find scc! visit successor:"<<successor<<"update LOW"<<LOW[index]<<std::endl;
        }
    }
    if(LOW[index] == DFN[index])//是该强连通分量子树的最小根，将其后的所有node出栈,保存得到的强连通分量
    {
        // std::cout<<"index:"<<index<<"scc root"<<std::endl;
        auto it = stack_subgraphs.end() - 1; 
        std:: vector<int> strongly_connected; 
        while(*it != index)
        {
            //std::cout<<*it<<"--";
            strongly_connected.insert(strongly_connected.begin(), *it);
            stack_subgraphs.pop_back();
            it = stack_subgraphs.end() - 1;
        }
        //std::cout<<*it;
        strongly_connected.insert(strongly_connected.begin(), *it);

        // std::cout<<"scc:";
        for(const auto& graph :strongly_connected)
        {
            // std::cout<<graph<<"~";
        }
        //std::cout<<std::endl;
        if(strongly_connected.size() > 1)
        {
            strongly_connected_subgraphs.push_back(strongly_connected);
        }
        stack_subgraphs.pop_back();//自身出栈
        //std::cout<<std::endl;
    }
}
std::vector<graph_adjacency_node> calculate_node_rank(
    std::vector<int>& strongly_connected, 
    std::vector<onnx::GraphProto>& Subgraphs,
    std::vector<onnx::GraphProto>& otherSubgraphs
    )
{
    onnx::GraphProto merged_graph;
    std::vector<graph_adjacency_node> node_rank_list;
    for(const auto& index : strongly_connected)
    {
        if(index < Subgraphs.size())
        {
            mergeGraphs(merged_graph, Subgraphs[index]);
        }
        else
        {
            mergeGraphs(merged_graph, otherSubgraphs[index - Subgraphs.size()]);
        }
    }
    int index = 0;
    for(const auto& node : merged_graph.node())
    {
        graph_adjacency_node node_rank;
        node_rank.name = node.name();
        node_rank.index = index;
        node_rank.rank = -1;
        node_rank_list.push_back(node_rank);
        index ++; 
    }
    int sort_count=0;
    int finished_flag=0;
    while(!finished_flag) 
    {
        finished_flag=1;
        int changed_sort_flag=0;
        if(sort_count==0)
        {
            changed_sort_flag=1;
            for(int i=0; i<merged_graph.node_size();i++) //遍历所有节点
            {
                int find_flag=0;
                for(const auto& input : merged_graph.node(i).input())
                {
                    for(int j=0; j<merged_graph.node_size();j++)
                    {
                        for(const auto& output : merged_graph.node(j).output())
                        {
                            if(input==output)
                            {
                                find_flag=1;
                                break;
                            }
                        }
                        if(find_flag){break;}
                    }
                    if(find_flag){break;}
                }
                if(!find_flag)
                {
                    node_rank_list[i].rank=sort_count;
                }
            }
            finished_flag=0;
        }
        else
        {
            for(int i=0; i<merged_graph.node_size();i++) 
            {
                int find_flag=0;
                if(node_rank_list[i].rank >= 0 && node_rank_list[i].rank < sort_count){continue;}////如果已经排过序了，跳过这个子图
                for(const auto& input : merged_graph.node(i).input())////遍历某个子图的所有input
                {
                    for(int j=0; j< merged_graph.node_size(); j++)////检查该子图的某个input是否是第j个子图的output
                    {
                        for(const auto& output : merged_graph.node(j).output())
                        {
                            if(output==input)
                            {
                                if((node_rank_list[j].rank < 0 || node_rank_list[j].rank >= sort_count))//若第j个子图尚未被排序
                                {
                                    find_flag=1;
                                    break;
                                }
                            }
                        }
                        if(find_flag){break;}

                    }
                    if(find_flag){break;}
                }
                if(!find_flag)
                {
                    node_rank_list[i].rank=sort_count;
                }
                else {node_rank_list[i].rank=sort_count+1;finished_flag=0;}
            }
        }
        sort_count++;
    }
    //若存在order = 0 node，则为master 要切割
    return node_rank_list;
}
std::vector<graph_adjacency_node> calculate_node_rank_v2(
    std::vector<int>& strongly_connected, 
    std::vector<onnx::GraphProto>& Subgraphs,
    std::vector<onnx::GraphProto>& otherSubgraphs,
    int subgraph_size,
    int other_subgraph_size
    )
{
    onnx::GraphProto merged_graph;
    std::vector<graph_adjacency_node> node_rank_list;
    for(const auto& index : strongly_connected)
    {
        if(index < subgraph_size)
        {
            mergeGraphs(merged_graph, Subgraphs[index]);
        }
        else
        {
            mergeGraphs(merged_graph, otherSubgraphs[index - subgraph_size]);
        }
    }
    int index = 0;
    for(const auto& node : merged_graph.node())
    {
        graph_adjacency_node node_rank;
        node_rank.name = node.name();
        node_rank.index = index;
        node_rank.rank = -1;
        node_rank_list.push_back(node_rank);
        index ++; 
    }
    int sort_count=0;
    int finished_flag=0;
    while(!finished_flag) 
    {
        finished_flag=1;
        int changed_sort_flag=0;
        if(sort_count==0)
        {
            changed_sort_flag=1;
            for(int i=0; i<merged_graph.node_size();i++) //遍历所有节点
            {
                int find_flag=0;
                for(const auto& input : merged_graph.node(i).input())
                {
                    for(int j=0; j<merged_graph.node_size();j++)
                    {
                        for(const auto& output : merged_graph.node(j).output())
                        {
                            if(input==output)
                            {
                                find_flag=1;
                                break;
                            }
                        }
                        if(find_flag){break;}
                    }
                    if(find_flag){break;}
                }
                if(!find_flag)
                {
                    node_rank_list[i].rank=sort_count;
                }
            }
            finished_flag=0;
        }
        else
        {
            for(int i=0; i<merged_graph.node_size();i++) 
            {
                int find_flag=0;
                if(node_rank_list[i].rank >= 0 && node_rank_list[i].rank < sort_count){continue;}////如果已经排过序了，跳过这个子图
                for(const auto& input : merged_graph.node(i).input())////遍历某个子图的所有input
                {
                    for(int j=0; j< merged_graph.node_size(); j++)////检查该子图的某个input是否是第j个子图的output
                    {
                        for(const auto& output : merged_graph.node(j).output())
                        {
                            if(output==input)
                            {
                                if((node_rank_list[j].rank < 0 || node_rank_list[j].rank >= sort_count))//若第j个子图尚未被排序
                                {
                                    find_flag=1;
                                    break;
                                }
                            }
                        }
                        if(find_flag){break;}

                    }
                    if(find_flag){break;}
                }
                if(!find_flag)
                {
                    node_rank_list[i].rank=sort_count;
                }
                else {node_rank_list[i].rank=sort_count+1;finished_flag=0;}
            }
        }
        sort_count++;
    }
    //若存在order = 0 node，则为master 要切割
    return node_rank_list;
}
void calculate_node_rank_v3(
    const onnx::GraphProto& merged_graph,
    std::vector<graph_adjacency_node>& node_rank_list
    )
{
    int index = 0;
    for(const auto& node : merged_graph.node())
    {
        graph_adjacency_node node_rank;
        node_rank.name = node.name();
        node_rank.index = index;
        node_rank.rank = -1;
        node_rank_list.push_back(node_rank);
        index ++; 
    }
    int sort_count=0;
    int finished_flag=0;
    while(!finished_flag) 
    {
        finished_flag=1;
        int changed_sort_flag=0;
        if(sort_count==0)
        {
            changed_sort_flag=1;
            for(int i=0; i<merged_graph.node_size();i++) //遍历所有节点
            {
                int find_flag=0;
                for(const auto& input : merged_graph.node(i).input())
                {
                    for(int j=0; j<merged_graph.node_size();j++)
                    {
                        for(const auto& output : merged_graph.node(j).output())
                        {
                            if(input==output)
                            {
                                find_flag=1;
                                break;
                            }
                        }
                        if(find_flag){break;}
                    }
                    if(find_flag){break;}
                }
                if(!find_flag)
                {
                    node_rank_list[i].rank=sort_count;
                }
            }
            finished_flag=0;
        }
        else
        {
            for(int i=0; i<merged_graph.node_size();i++) 
            {
                int find_flag=0;
                if(node_rank_list[i].rank >= 0 && node_rank_list[i].rank < sort_count){continue;}////如果已经排过序了，跳过这个子图
                for(const auto& input : merged_graph.node(i).input())////遍历某个子图的所有input
                {
                    for(int j=0; j< merged_graph.node_size(); j++)////检查该子图的某个input是否是第j个子图的output
                    {
                        for(const auto& output : merged_graph.node(j).output())
                        {
                            if(output==input)
                            {
                                if((node_rank_list[j].rank < 0 || node_rank_list[j].rank >= sort_count))//若第j个子图尚未被排序
                                {
                                    find_flag=1;
                                    break;
                                }
                            }
                        }
                        if(find_flag){break;}

                    }
                    if(find_flag){break;}
                }
                if(!find_flag)
                {
                    node_rank_list[i].rank=sort_count;
                }
                else {node_rank_list[i].rank=sort_count+1;finished_flag=0;}
            }
        }
        sort_count++;
    }
}
int get_cut_rank(std::vector<graph_adjacency_node>& scc_node_rank)
{
    int min_cut_rank = -1;
    //get min
    for(int i=0; i<scc_node_rank.size(); i++)
    {
        if(scc_node_rank[i].rank <min_cut_rank || min_cut_rank < 0)
        {
            min_cut_rank = scc_node_rank[i].rank;
        }
    }
    int find_flag = 1;
    while(find_flag)
    {
        min_cut_rank ++;
        int temp_find_flag = 0;
        for(int i=0; i<scc_node_rank.size(); i++)
        {
            if(scc_node_rank[i].rank == min_cut_rank)
            {
                temp_find_flag = 1;
                break;
            }
        }
        find_flag = temp_find_flag;
    }
    return min_cut_rank;
}
std::vector<int> get_cut_rank_v2(std::vector<graph_adjacency_node>& scc_node_rank)
{
    std::vector<int> cut_rank_list;
    int min_cut_rank = -1;
    int max_rank = 0;
    //get min
    for(int i=0; i<scc_node_rank.size(); i++)
    {
        if(scc_node_rank[i].rank <min_cut_rank || min_cut_rank < 0)
        {
            min_cut_rank = scc_node_rank[i].rank;
        }
        if(scc_node_rank[i].rank >max_rank)
        {
            max_rank = scc_node_rank[i].rank;
        }
    }
    int find_flag = 1;
    while(find_flag)
    {
        min_cut_rank ++;
        int temp_find_flag = 0;
        for(int i=0; i<scc_node_rank.size(); i++)
        {
            if(scc_node_rank[i].rank == min_cut_rank)
            {
                temp_find_flag = 1;
                break;
            }
        }
        find_flag = temp_find_flag;
    }
    cut_rank_list.push_back(min_cut_rank);
    int new_cut_rank = 1;
    int cut_rank = min_cut_rank;
    while (cut_rank < max_rank)
    {
        cut_rank = cut_rank + 1;
        int rank_flag = 0;
        int rank_plus_flag = 0;
        for(int i=0; i<scc_node_rank.size(); i++)
        {
            if(scc_node_rank[i].rank == cut_rank)
            {
                rank_flag = 1;
            }
            else if(scc_node_rank[i].rank == cut_rank + 1)
            {
                rank_plus_flag = 1;
            }
        }
        if(rank_flag == 0&& rank_plus_flag ==1)
        {
            cut_rank_list.push_back(cut_rank + 1);
        }
    }

    return cut_rank_list;
}
void eliminate_scc(
    std::vector<std::vector<int>>& strongly_connected_subgraphs, 
    std::vector<onnx::GraphProto>& Subgraphs,
    std::vector<onnx::GraphProto>& otherSubgraphs
)
{
    int subgraph_size = Subgraphs.size();
    int other_subgraph_size = otherSubgraphs.size();
    for(auto& strongly_connected : strongly_connected_subgraphs)
    {
        std::vector<graph_adjacency_node> node_rank_list = calculate_node_rank_v2(strongly_connected, Subgraphs, otherSubgraphs,subgraph_size, other_subgraph_size);
        int index_all = 0;
        for(const auto scc_index : strongly_connected)
        {
            onnx::GraphProto scc_graph;
            if(scc_index < subgraph_size)
            {
                scc_graph = Subgraphs[scc_index];
            }
            else
            {
                scc_graph = otherSubgraphs[scc_index - subgraph_size];
            }
            //onnx::GraphProto scc_graph = determinegraphtype(scc_index, Subgraphs, otherSubgraphs);
            std::vector<graph_adjacency_node> scc_node_rank;
            for(int i=0; i< scc_graph.node_size(); i++)
            {
                scc_node_rank.push_back(node_rank_list[index_all]);
                index_all++;
            }
            int cut_rank = get_cut_rank(scc_node_rank);
            onnx::GraphProto temp_graph_upper;
            onnx::GraphProto temp_graph_lower;
            for(int i=0; i<scc_graph.node_size(); i++)
            {
                if(scc_node_rank[i].rank < cut_rank)
                {
                    *temp_graph_upper.add_node() = scc_graph.node(i);
                }
                else
                {
                    *temp_graph_lower.add_node() = scc_graph.node(i);
                }
            }
            std::cout<<"scc size: "<<scc_graph.node_size()<<std::endl;
            std::cout<<"upper graph size:"<<temp_graph_upper.node_size()<<std::endl;
            std::cout<<"lower graph size:"<<temp_graph_lower.node_size()<<std::endl;
            if(scc_index < subgraph_size)
            {
                Subgraphs[scc_index] = temp_graph_upper;
                if(temp_graph_lower.node_size()>0)
                {
                    Subgraphs.push_back(temp_graph_lower);
                    std::cout<<"pushed"<<std::endl;
                }
                
            }
            else
            {
                otherSubgraphs[scc_index - subgraph_size] = temp_graph_upper;
                if(temp_graph_lower.node_size()>0)
                {
                    otherSubgraphs.push_back(temp_graph_lower);
                    std::cout<<"pushed"<<std::endl;
                }
            }
        }
    }
}
void eliminate_scc_v2(
    std::vector<std::vector<int>>& strongly_connected_subgraphs, 
    std::vector<onnx::GraphProto>& Subgraphs,
    std::vector<onnx::GraphProto>& otherSubgraphs,
    const onnx::GraphProto& g
)
{
    int subgraph_size = Subgraphs.size();
    int other_subgraph_size = otherSubgraphs.size();
    for(auto& strongly_connected : strongly_connected_subgraphs)
    {
        std::vector<graph_adjacency_node> node_rank_list;
        calculate_node_rank_v3(g, node_rank_list);
        int index_all = 0;
        for(const auto scc_index : strongly_connected)
        {
            onnx::GraphProto scc_graph;
            if(scc_index < subgraph_size)
            {
                scc_graph = Subgraphs[scc_index];
            }
            else
            {
                scc_graph = otherSubgraphs[scc_index - subgraph_size];
            }
            //onnx::GraphProto scc_graph = determinegraphtype(scc_index, Subgraphs, otherSubgraphs);
            std::vector<graph_adjacency_node> scc_node_rank;
            for(int i=0; i< scc_graph.node_size(); i++)
            {
                // scc_node_rank.push_back(node_rank_list[index_all]);
                // index_all++;
                for(int j = 0; j < node_rank_list.size(); j++)
                {
                    if(scc_graph.node(i).name() == node_rank_list[j].name)
                    {
                        scc_node_rank.push_back(node_rank_list[j]);
                        break;
                    }
                }
            }
            std::vector<int> cut_rank = get_cut_rank_v2(scc_node_rank);
            onnx::GraphProto temp_graph_upper;
            for(int i=0; i<scc_graph.node_size(); i++)
            {
                if(scc_node_rank[i].rank < cut_rank[0])
                {
                    *temp_graph_upper.add_node() = scc_graph.node(i);
                }
            }
            std::cout<<"scc index"<<scc_index<<" scc size: "<<scc_graph.node_size()<<std::endl;
            std::cout<<"scc node rank: ";
            for(int i=0; i< scc_graph.node_size(); i++)
            {
                std::cout<<scc_node_rank[i].name<<" "<<scc_node_rank[i].rank<<" ";
            }
            std::cout<<std::endl;
            std::cout<<"upper graph size:"<<temp_graph_upper.node_size()<<std::endl;
            // std::cout<<"lower graph size:"<<temp_graph_lower.node_size()<<std::endl;
            if(scc_index < subgraph_size)
            {
                Subgraphs[scc_index] = temp_graph_upper;
            }
            else
            {
                otherSubgraphs[scc_index - subgraph_size] = temp_graph_upper;
            }
            for(int i = 0; i< cut_rank.size() -1;  i++)
            {
                onnx::GraphProto temp_graph_lower;
                for(int j=0; j<scc_graph.node_size(); j++)
                {
                    if(scc_node_rank[j].rank >= cut_rank[i]&& scc_node_rank[j].rank < cut_rank[i+1])
                    {
                        *temp_graph_lower.add_node() = scc_graph.node(j);
                    }
                }
                // std::cout<<"cut rank"<<cut_rank[i]<< "graph size:"<<temp_graph_lower.node_size();
                // for(int i=0; i< temp_graph_lower.node_size(); i++)
                // {
                //     std::cout<<temp_graph_lower.node(i).name()<<" ";
                // }
                // std::cout<<std::endl;
                if(scc_index < subgraph_size)
                {
                    if(temp_graph_lower.node_size()>0)
                    {
                        Subgraphs.push_back(temp_graph_lower);
                    }
                    
                }
                else
                {
                    if(temp_graph_lower.node_size()>0)
                    {
                        otherSubgraphs.push_back(temp_graph_lower);
                    }
                }
            }
            onnx::GraphProto temp_graph_lower;
            for(int j=0; j<scc_graph.node_size(); j++)
            {
                if(scc_node_rank[j].rank >= cut_rank[cut_rank.size() -1])
                {
                    *temp_graph_lower.add_node() = scc_graph.node(j);
                }
            }
            // std::cout<<"cut rank"<<cut_rank[cut_rank.size() -1]<< "graph size:"<<temp_graph_lower.node_size();
            // for(int i=0; i< temp_graph_lower.node_size(); i++)
            // {
            //     std::cout<<temp_graph_lower.node(i).name()<<" ";
            // }
            // std::cout<<std::endl;
            if(scc_index < subgraph_size)
            {
                if(temp_graph_lower.node_size()>0)
                {
                    Subgraphs.push_back(temp_graph_lower);
                }
                
            }
            else
            {
                if(temp_graph_lower.node_size()>0)
                {
                    otherSubgraphs.push_back(temp_graph_lower);
                }
            }
        }
    }
}
onnx::GraphProto determinegraphtype(
    int index,
    std::vector<onnx::GraphProto>& Subgraphs,
    std::vector<onnx::GraphProto>& otherSubgraphs
)
{
    if(index < Subgraphs.size())
    {
        return Subgraphs[index];
    }
    else
    {
        return otherSubgraphs[index - Subgraphs.size()];
    }
}
void find_subgraph_pair(
    std::vector<std::vector<int>>& strongly_connected_subgraphs, 
    std::vector<onnx::GraphProto>& Subgraphs,
    std::vector<onnx::GraphProto>& otherSubgraphs,
    std::vector<std::unordered_set<NodeTensor>>& graphs_inputs,
    std::vector<std::unordered_set<NodeTensor>>& graphs_outputs,
    std::vector<std::vector<std::vector<int>>>& sccs_pairs
)
{
    int count = 0;
    for(const auto& strongly_connected :strongly_connected_subgraphs)
    {
        std::vector<onnx::GraphProto> scc_graphs;
        std::vector<std::unordered_set<NodeTensor>> scc_graphs_inputs;
        std::vector<std::unordered_set<NodeTensor>> scc_graphs_outputs;
        for(const auto& index : strongly_connected)
        {
            std::unordered_set<NodeTensor> graph_inputs = graphs_inputs[index];
            std::unordered_set<NodeTensor> graph_outputs = graphs_outputs[index];
            //onnx::GraphProto subgraph = determinegraphtype(index, Subgraphs, otherSubgraphs);
            //scc_graphs.push_back(subgraph);
            //
            scc_graphs_inputs.push_back(graph_inputs);
            scc_graphs_outputs.push_back(graph_outputs);
        }
        int find_flag=0;
        std::vector<std::vector<int>> scc_pairs;
        for(int i = 0; i < strongly_connected.size(); i++)
        {
            std::vector<int> is_pushed;
            for(int j = 0; j < strongly_connected.size(); j++)
            {
                is_pushed.push_back(0);
            }
            for(const auto& graph_input : scc_graphs_inputs[i])
            {
                
                for(int j = i + 1; j < strongly_connected.size(); j++)
                {
                    std::vector<int> scc_pair;
                    if(scc_graphs_outputs[j].find(graph_input)!=scc_graphs_outputs[j].end()&& is_pushed[j]==0)
                    {
                        for(const auto& graph_output : scc_graphs_outputs[i])
                        {
                            if(scc_graphs_inputs[j].find(graph_output)!=scc_graphs_inputs[j].end())
                            {
                                scc_pair.push_back(strongly_connected[i]);
                                scc_pair.push_back(strongly_connected[j]);
                                scc_pairs.push_back(scc_pair);
                                is_pushed[j]=1;
                                find_flag=1;
                                break;
                            }
                        }
                    }
                }
            }
        }
        if(scc_pairs.size() != 0)
        {
            sccs_pairs.push_back(scc_pairs);
        }
    count ++;
    }
    
    //////////////
    for(const auto& scc_pairs : sccs_pairs)
    {
        std::cout << "scc pair:";
        for(const auto& scc_pair : scc_pairs)
        {
            
            for(const auto& scc_id : scc_pair)
            {
                std::cout << scc_id << " ";
            }
            std::cout << ";";
        }
        std::cout << std::endl;
    }
}
std::vector<int> cut_pair(
    std::vector<onnx::GraphProto>& Subgraphs,
    std::vector<onnx::GraphProto>& otherSubgraphs,
    std::vector<std::unordered_set<NodeTensor>>& graphs_inputs,
    std::vector<std::unordered_set<NodeTensor>>& graphs_outputs,
    std::vector<int>& scc_pair,
    std::vector<onnx::GraphProto>& scc_pair_cut
)
{
    std::vector<graph_adjacency_node> pair_node_list = calculate_node_rank(scc_pair, Subgraphs,otherSubgraphs);
    int master_graph = 0;
    for(const auto& node : pair_node_list)
    {
        if(node.rank==0)
        {
            int find_flag = -1;
            onnx::GraphProto graph_temp = determinegraphtype(scc_pair[0],Subgraphs, otherSubgraphs);
            for(const auto& graph_node : graph_temp.node())
            {
                if(graph_node.name()==node.name)
                {
                    find_flag = 1;
                    break;
                }
            }
            if(find_flag == 1)
            {
                master_graph = 0;
                //std::cout<<"master_graph: "<<scc_pair[master_graph]<<std::endl;
                break;
            }
            else{master_graph = 1;break;}
        }
    }
    int slave_graph = 1 - master_graph;
    //找到master与slave图相接的位置
    int cut_rank = -1;
    // for(const auto& node : pair_node_list)
    // {
    //     std::cout<<node.rank<<" ";
    // }
    // std::cout<<std::endl;
    for(const auto& output : graphs_outputs[scc_pair[slave_graph]])
    {
        //std::cout<<output.name<<std::endl;
        for(const auto& input : graphs_inputs[scc_pair[master_graph]])
        {
            
            if(input.name ==output.name)
            {
                //std::cout<<"find!!!"<<std::endl;
                int node_index = 0;
                onnx::GraphProto graph_temp = determinegraphtype(scc_pair[slave_graph],Subgraphs, otherSubgraphs);
                for(const auto& graph_node : graph_temp.node())
                {
                    int update_node_rank = 0;
                    for(const auto& output_node : graph_node.output())
                    {
                        if(output_node==output.name)
                        {
                            if(slave_graph==0)
                            {
                                if(cut_rank==-1||cut_rank>pair_node_list[node_index].rank)
                                {
                                    cut_rank = pair_node_list[node_index].rank; 
                                    //std::cout<<"cut_rank update: "<<cut_rank<<std::endl;
                                }
                            }
                            else
                            {
                                onnx::GraphProto graph_temp_1 = determinegraphtype(scc_pair[master_graph], Subgraphs, otherSubgraphs);
                                if(cut_rank==-1||cut_rank>pair_node_list[node_index+ graph_temp_1.node_size()].rank)
                                {
                                    cut_rank = pair_node_list[node_index+ graph_temp_1.node_size()].rank; 
                                    //std::cout<<"cut_rank update: "<<cut_rank<<std::endl;
                                }
                                //else{std::cout<<"cut_rank not update: "<<pair_node_list[node_index+ graph_temp_1.node_size()].rank<<std::endl;}
                            }
                            update_node_rank = 1;
                            break;
                        }
                    }
                    if(update_node_rank == 1)
                    {
                        break;
                    }
                    node_index++;
                }
                break;
            }
        }
    }
    //按照cut rank切master
    onnx::GraphProto master_upper;
    onnx::GraphProto master_lower;
    int node_index = 0;
    onnx::GraphProto graph_temp = determinegraphtype(scc_pair[master_graph],Subgraphs, otherSubgraphs);
    int master_graph_size = graph_temp.node_size();
    //std::cout << "master_graph size: " << graph_temp.node_size() << std::endl;
    int slave_graph_size;
    for(const auto& node : graph_temp.node())
    {
        int node_rank;
        if(master_graph == 0)
        {
            node_rank = pair_node_list[node_index].rank;
        }
        else
        {
            onnx::GraphProto graph_temp_2 = determinegraphtype(scc_pair[slave_graph],Subgraphs, otherSubgraphs);
            node_rank = pair_node_list[node_index+ graph_temp_2.node_size()].rank;
        }
        if(node_rank<cut_rank)
        {
            *master_upper.add_node() = node;
        }
        else
        {
            *master_lower.add_node() = node;
        }
        node_index++;
    }
    scc_pair_cut.push_back(master_upper);
    scc_pair_cut.push_back(master_lower);
    scc_pair_cut.push_back(determinegraphtype(scc_pair[slave_graph],Subgraphs, otherSubgraphs));
    if(master_graph == 1)
    {
        int temp = scc_pair[0];
        scc_pair[0] = scc_pair[1];
        scc_pair[1] = temp;
        master_graph = 0;
    }
    std::vector<int> return_value;
    return_value.push_back(master_graph);
    return_value.push_back(cut_rank);
    return return_value;
}
void cut_crossed_pair(
    std::vector<onnx::GraphProto>& Subgraphs,
    std::vector<onnx::GraphProto>& otherSubgraphs,
    std::vector<std::unordered_set<NodeTensor>>& graphs_inputs,
    std::vector<std::unordered_set<NodeTensor>>& graphs_outputs,
    std::vector<std::vector<int>> scc_pairs_crossed,
    std::vector<std::vector<onnx::GraphProto>>& scc_pair_cut_final
)
{
    int pair_index = 0;
    int cut_rank_record = -1;
    int index = 0;
    int crossed_index;
    if(scc_pairs_crossed[0][0] == scc_pairs_crossed[1][0] || scc_pairs_crossed[0][0] == scc_pairs_crossed[1][1])
        {
            crossed_index = scc_pairs_crossed[0][0];
        }
    else
        {
            crossed_index = scc_pairs_crossed[0][1];
        }
    for(auto& scc_pair : scc_pairs_crossed)
    {
        std::vector<onnx::GraphProto> scc_pair_cut;
        std::vector<int> return_value = cut_pair(Subgraphs, otherSubgraphs, graphs_inputs, graphs_outputs, scc_pair, scc_pair_cut);  
        scc_pair_cut_final.push_back(scc_pair_cut);
        if(scc_pair[return_value[0]] == crossed_index)
        {
            std::cout<<"master graph crossed"<<std::endl;
        }
        else
        {
            std::cout<<"error : slave graph crossed"<<std::endl;
            std::exit(1);
        }        
        if(return_value[1] < cut_rank_record || cut_rank_record == -1)
        {
            cut_rank_record = return_value[1];
            pair_index = index;
        }
        index ++;      
    }
    //若非要保存的，删除master相关图
    for(int i = 0; i < scc_pair_cut_final.size(); i++)
    {
        if(i != pair_index)
        {
            scc_pair_cut_final[i].erase(scc_pair_cut_final[i].begin());
            scc_pair_cut_final[i].erase(scc_pair_cut_final[i].begin());
        }
    }
    int size_after_cut= 0;
}
void find_crossed_pair(
    std::vector<std::vector<int>> & scc_pairs,
    std::vector<std::vector<std::vector<int>>> & scc_crossed_pairs_multi
)
{
    //std::cout<< "find cross begin"<<std::endl;
    std::vector<int> pushed;
    for(int i = 0; i < scc_pairs.size(); i++)
    {
        pushed.push_back(0);
    }
    for(int i = 0; i < scc_pairs.size(); i++)
    {
        std::vector<std::vector<int>> scc_crossed_pairs;
        scc_crossed_pairs.push_back(scc_pairs[i]);
        for(int j = i + 1; j < scc_pairs.size(); j++)
        {
            if(pushed[j] == 0)
            {
                if(scc_pairs[i][0] == scc_pairs[j][1] || scc_pairs[i][1] == scc_pairs[j][0] || scc_pairs[i][0] == scc_pairs[j][0] || scc_pairs[i][1] == scc_pairs[j][1])
               {
                    scc_crossed_pairs.push_back(scc_pairs[j]);
                    pushed[j] = 1;
               }
            }
            ///////////////判断用，避免出现pair chain
            int crossed_index;
            if(scc_crossed_pairs.size() > 1)
            {
                if(scc_crossed_pairs[0][0] == scc_crossed_pairs[1][0] || scc_crossed_pairs[0][0] == scc_crossed_pairs[1][1])
                {
                    crossed_index = scc_crossed_pairs[0][0];
                }
                else
                {
                    crossed_index = scc_crossed_pairs[0][1];
                }
                if(crossed_index != scc_pairs[j][0] && crossed_index != scc_pairs[j][1])
                {
                    std::cout<<"error : find pair chain"<<std::endl;
                    std::exit(1);
                }
            }
            /////////
        }
        if(scc_crossed_pairs.size()> 1)
        {
            pushed[i] = 1;
            scc_crossed_pairs_multi.push_back(scc_crossed_pairs);
        }
    }
    //分离出crossed pair
    for(int i = scc_pairs.size() - 1;i >= 0; i--)
    {
        if(pushed[i] == 1)
        {
            scc_pairs.erase(scc_pairs.begin() + i);
        }
    }
    for(const auto& scc_crossed_pairs : scc_crossed_pairs_multi)
    {
        std::cout<<"crossed pair: ";
        for(const auto& scc_pairs : scc_crossed_pairs)
        {
            for(const auto& scc_elem : scc_pairs)
            {
                std::cout<<scc_elem<<" ";
            }
            std::cout<<";";
        }
        std::cout<<std::endl;
    }
}
void eliminate_pair(
    std::vector<onnx::GraphProto>& Subgraphs,
    std::vector<onnx::GraphProto>& otherSubgraphs,
    std::vector<std::unordered_set<NodeTensor>>& graphs_inputs,
    std::vector<std::unordered_set<NodeTensor>>& graphs_outputs,
    std::vector<std::vector<int>>& strongly_connected_subgraphs
)
{
    std::vector<std::vector<std::vector<onnx::GraphProto>>> sccs_pairs_cut_multi;
    std::vector<std::vector<std::vector<int>>> sccs_pairs;
    find_subgraph_pair(strongly_connected_subgraphs, Subgraphs, otherSubgraphs, graphs_inputs, graphs_outputs, sccs_pairs);
    for(auto &scc_pairs : sccs_pairs)
    {
        // std::cout<<"scc pairs size before cut: "<<scc_pairs.size()<<std::endl;
        std::vector<std::vector<onnx::GraphProto>> scc_pairs_cut;
        std::vector<std::vector<std::vector<int>>> scc_crossed_pairs_multi;
        find_crossed_pair(scc_pairs, scc_crossed_pairs_multi);
        for(auto &scc_pair : scc_pairs)
        {
            std::vector<onnx::GraphProto> scc_pair_cut;
            cut_pair(Subgraphs, otherSubgraphs, graphs_inputs, graphs_outputs, scc_pair, scc_pair_cut);
            scc_pairs_cut.push_back(scc_pair_cut);
        }
        for(auto &scc_crossed_pairs : scc_crossed_pairs_multi)
        {
            std::vector<std::vector<onnx::GraphProto>> scc_pairs_cut_multi;
            cut_crossed_pair(Subgraphs, otherSubgraphs, graphs_inputs, graphs_outputs, scc_crossed_pairs, scc_pairs_cut_multi);
            scc_pairs_cut.insert(scc_pairs_cut.end(),scc_pairs_cut_multi.begin(), scc_pairs_cut_multi.end());
            scc_pairs.insert(scc_pairs.end(),scc_crossed_pairs.begin(), scc_crossed_pairs.end());
        }
        sccs_pairs_cut_multi.push_back(scc_pairs_cut);
        // std::cout<<"scc pairs size after cut: "<<scc_pairs.size()<<std::endl;
        // std::cout<<"scc pairs graph size after cut: "<<scc_pairs_cut.size()<<std::endl;
    }
    int subgraph_size = Subgraphs.size(); 
    int other_subgraph_size = otherSubgraphs.size();
    std::cout<<"size"<<subgraph_size<<" "<<other_subgraph_size<<std::endl;
    for(int i = sccs_pairs.size() - 1; i >= 0; i--)
    {
        for(int j = sccs_pairs[i].size() - 1; j >= 0 ; j--)
        {
            //std::cout<<"scc pair size"<<sccs_pairs_cut_multi[i][j].size()<<std::endl;
            if(sccs_pairs_cut_multi[i][j].size() > 1)
            {         
                      
                if(sccs_pairs[i][j][0] < subgraph_size)
                {
                    // std::cout<<sccs_pairs[i][j][0]<<" size of original subgraph:"<<Subgraphs[sccs_pairs[i][j][0]].node_size()<<std::endl;
                    // std::cout<<sccs_pairs[i][j][1] - subgraph_size<<" size of original subgraph:"<<otherSubgraphs[sccs_pairs[i][j][1] - subgraph_size].node_size()<<std::endl;
                    // std::cout<<"size of cut subgraph:"<<sccs_pairs_cut_multi[i][j][0].node_size()<<" "<<sccs_pairs_cut_multi[i][j][1].node_size()<<std::endl;
                    Subgraphs[sccs_pairs[i][j][0]] = sccs_pairs_cut_multi[i][j][0];
                    Subgraphs.push_back(sccs_pairs_cut_multi[i][j][1]);
                }
                else
                {
                    // std::cout<<"other"<<sccs_pairs[i][j][0] - subgraph_size<<" size of original subgraph:"<<otherSubgraphs[sccs_pairs[i][j][0] - subgraph_size].node_size()<<std::endl;
                    // std::cout<<sccs_pairs[i][j][1]<<" size of original subgraph:"<<Subgraphs[sccs_pairs[i][j][1]].node_size()<<std::endl;
                    // std::cout<<"size of cut subgraph:"<<sccs_pairs_cut_multi[i][j][0].node_size()<<" "<<sccs_pairs_cut_multi[i][j][1].node_size()<<std::endl;
                    otherSubgraphs[sccs_pairs[i][j][0] - subgraph_size] = sccs_pairs_cut_multi[i][j][0];
                    otherSubgraphs.push_back(sccs_pairs_cut_multi[i][j][1]);
                }
            }
            //std::cout<<"succeed"<<std::endl;
        }
    }
}
void eliminate_connection(
    std::vector<std::vector<int>>& strongly_connected_subgraphs, 
    std::vector<onnx::GraphProto>& Subgraphs,
    std::vector<onnx::GraphProto>& otherSubgraphs,
    std::vector<std::unordered_set<NodeTensor>>& graphs_outputs)
{   
    std::vector<onnx::GraphProto> subgraphs_all;
    subgraphs_all.insert(subgraphs_all.end(), Subgraphs.begin(),Subgraphs.end());
    subgraphs_all.insert(subgraphs_all.end(), otherSubgraphs.begin(),otherSubgraphs.end());
    std::vector<onnx::GraphProto> lower_subgraphs_all;
    std::vector<onnx::GraphProto> lower_othersubgraphs_all;
    for(const auto& strongly_connected :strongly_connected_subgraphs)
    {
        std::vector<onnx::GraphProto> upper_subgraphs;
        std::vector<onnx::GraphProto> lower_subgraphs;
        for(const auto& graph_index : strongly_connected)
        {
            onnx::GraphProto upper_subgraph_single;
            onnx::GraphProto lower_subgraph_single;
            std::vector<onnx::NodeProto> upper_nodeset_single;
            for(const auto& node : subgraphs_all[graph_index].node())
            {
                int find_flag = 0;
                for(const auto& input : node.input())
                {
                    for(const auto& graph_index_out : strongly_connected)//是否有来自该强连通分量其他子图的输入或来自其他lower node的输入
                    {
                        if(graph_index_out == graph_index)//避免计入本子图
                        {
                            continue;
                        }
                        for(const auto& output : graphs_outputs[graph_index_out])
                        {
                            if(output.name == input)
                            {
                                find_flag = 1;
                                break;
                            }
                        }
                        if(find_flag == 1)
                        {
                            break;
                        }
                    }
                    if(find_flag == 1)
                    {
                        break;
                    }
                }
                if(find_flag == 1)
                {
                    *lower_subgraph_single.add_node() = node;
                }
                else upper_nodeset_single.push_back(node);
                //第一阶lower_node的后继和后继的后继也是lower_node，其他的是uppernode，待完善
            }
            int find_new_lower = 1;
            while(find_new_lower)
            {
                int find_new_lower_temp = 0;
                std::vector<int> is_lower;
                for(const auto& node : upper_nodeset_single)
                {
                    is_lower.push_back(0);
                    for(const auto& input : node.input())
                    {
                        for(const auto& node_lower : lower_subgraph_single.node())
                        {
                            for(const auto & output_lower : node_lower.output())
                            {
                                if(input == output_lower)
                                {
                                    find_new_lower_temp = 1;
                                    *lower_subgraph_single.add_node() = node;
                                    is_lower.pop_back();
                                    is_lower.push_back(1);
                                    break;
                                }
                            }
                            if(is_lower.back() == 1)
                            {
                                break;
                            }
                        }
                        if(is_lower.back() == 1)
                        {
                            break;
                        }
                    }
                }
                for(int i = is_lower.size()-1; i >= 0; --i)
                {
                    if(is_lower[i] == 1)
                    {
                        upper_nodeset_single.erase(upper_nodeset_single.begin() + i);
                    }
                }
                find_new_lower = find_new_lower_temp;
            }
            for(const auto & node : upper_nodeset_single)
            {
                *upper_subgraph_single.add_node() = node;
            }
            // if(upper_nodeset_single.size() == 0)
            // {
            //     std::cout<<"warning: upper subgraph is empty"<<std::endl;
            // }
            upper_subgraphs.push_back(upper_subgraph_single);
            lower_subgraphs.push_back(lower_subgraph_single);
        }
        //接下来的问题：如何把强连通分量和子图更新的信息传递到原始图中，并更新先后关系
        //upper保存到原位置，lower放到最后 lower是会改变子图长度的，最后再更新,但是如何确定lower的类型？
        int i = 0;
        for(const auto& graph_index : strongly_connected)
        {
            if(upper_subgraphs[i].node_size() != 0)
            {
                if(Subgraphs.size() > graph_index)
                {
                    Subgraphs[graph_index] = upper_subgraphs[i];
                    lower_subgraphs_all.push_back(lower_subgraphs[i]);
                }
                else
                {
                    otherSubgraphs[graph_index - Subgraphs.size()] = upper_subgraphs[i];
                    if(lower_subgraphs[i].node_size() != 0)
                    {
                        lower_othersubgraphs_all.push_back(lower_subgraphs[i]);
                    }
                    
                }
            }
            i++;
        }
    }
    Subgraphs.insert(Subgraphs.end(), lower_subgraphs_all.begin(), lower_subgraphs_all.end());
    otherSubgraphs.insert(otherSubgraphs.end(), lower_othersubgraphs_all.begin(), lower_othersubgraphs_all.end());
}

//////////////
///////
// Function to find other subgraphs in the original graph
std::vector<onnx::GraphProto> findOtherSubgraphs(const onnx::GraphProto& originalGraph, 
                                                 const std::vector<onnx::GraphProto>& knownSubgraphs) {
    std::vector<onnx::GraphProto> otherSubgraphs;

    // Create a set to store node names in the known subgraphs
    std::set<std::string> knownSubgraphNodeNames;
////获取已知子图和原图的node名称序列
    // Add node names from known subgraphs to the set
    for (const auto& subgraph : knownSubgraphs) {
        for (const auto& node : subgraph.node()) {
            knownSubgraphNodeNames.insert(node.name());
        }
    }

    // Create a set to store all node names in the original graph
    std::set<std::string> originalGraphNodeNames;
    for (const auto& node : originalGraph.node()) {
        originalGraphNodeNames.insert(node.name());
    }

    // Iterate over nodes in the original graph and find other subgraphs
    int startIndex = 0;
    int endIndex = -1;
    for (int i = 0; i < originalGraph.node_size(); ++i) {
        // If the current node name is not in the set of known subgraph node names, it belongs to a new subgraph////寻找不在已知子图中的node，将其记为endIndex
        if (knownSubgraphNodeNames.find(originalGraph.node(i).name()) == knownSubgraphNodeNames.end()) {
            // Determine the end index of the current subgraph
            endIndex = i;
        } else {
            // Create a new subgraph and add nodes from startIndex to endIndex////如果找到了已知子图中的node，并且它的前面存在未知子图中的node，那就对之前的node做总结，创立未知子图
            if (endIndex >= startIndex) {
                onnx::GraphProto newSubgraph;
                for (int j = startIndex; j <= endIndex; ++j) {
                    *newSubgraph.add_node() = originalGraph.node(j);
                }
                otherSubgraphs.push_back(newSubgraph);
            }

            // Update the startIndex for the next subgraph
            startIndex = i + 1;
        }
    }

    // Create the last subgraph (if any) from startIndex to the end of the original graph
    if (startIndex < originalGraph.node_size()) {
        onnx::GraphProto lastSubgraph;
        for (int j = startIndex; j < originalGraph.node_size(); ++j) {
            *lastSubgraph.add_node() = originalGraph.node(j);
        }
        otherSubgraphs.push_back(lastSubgraph);
    }

    return otherSubgraphs;
}

std::vector<onnx::GraphProto> processNpuSubgraphs(std::vector<onnx::GraphProto>& Subgraphs_, const std::vector<std::string>& NPUSupportOp, std::vector<std::string> NPUPreferOp) {
    // 创建一个新的向量来存储包含 NPUSupportOp 的子图
    std::vector<onnx::GraphProto> NPUSubgraphs;
    std::vector<onnx::GraphProto> CPUSubgraphs;

    // 遍历每个子图
    for (size_t i = 0; i < Subgraphs_.size(); ++i) {
        auto subgraph = Subgraphs_[i];

        // 遍历当前子图中的每个节点
        int npustartIndex = 0;
        int npuendIndex = -1;
        int cpustartIndex = 0;
        int cpuendIndex = -1;

        for (int j = 0; j < subgraph.node_size(); ++j) {
            const onnx::NodeProto& node = subgraph.node(j);

            // 检查当前节点的操作是否在 NPUSupportOp 中
            if (std::find(NPUSupportOp.begin(), NPUSupportOp.end(), node.op_type()) != NPUSupportOp.end()) {
                if (cpuendIndex >= cpustartIndex) {
                    onnx::GraphProto newSubgraph;
                    for (int k = cpustartIndex; k <= cpuendIndex; ++k) {
                        *newSubgraph.add_node() = subgraph.node(k);
                    }
                    CPUSubgraphs.push_back(newSubgraph);
                }
                npuendIndex = j;
                cpustartIndex = j + 1;
            } else {
                if (npuendIndex >= npustartIndex) {
                    // 判断这一部分在NPU上执行是否有意义
                    int flag = 0;
                    onnx::GraphProto newSubgraph;
                    for (int k = npustartIndex; k <= npuendIndex; ++k) {
                        // 在 vector 中查找目标字符串
                        auto it = std::find(NPUPreferOp.begin(), NPUPreferOp.end(), subgraph.node(k).op_type());
                        if (it != NPUPreferOp.end()) {
                            flag = 1;
                        }
                        *newSubgraph.add_node() = subgraph.node(k);
                    }

                    // 1.这个NPU子图中存在可以大幅度并行加速的算子（包含NPU only算子），合法
                    // 2.这个NPU子图没有可以大幅加速的算子，但是如果足够大,量变引起质变，合法
                    //      .2分许这个字图输入输出需要传输的数据量，如果数据量大，则可以往前或者往后挪一两个
                    
                    // 1.这个NPU子图中存在可以大幅度并行加速的算子
                    if (flag) {
                        // 查询子图的输入输出是否可以微调，主要是为了减少传输
                        if (npustartIndex != 0) {
                            // 记录当前输入节点的input size

                        }
                        if (npustartIndex < subgraph.node_size() - 1) {
                            /* code */
                        }

                        NPUSubgraphs.push_back(newSubgraph);
                    } else if (npuendIndex - npustartIndex + 1 > 10) { //TODO这里有待商榷
                        NPUSubgraphs.push_back(newSubgraph);
                    } else {
                        CPUSubgraphs.push_back(newSubgraph);
                    }
                }
                cpuendIndex = j;
                npustartIndex = j + 1;
            }
        }

        if (npustartIndex < subgraph.node_size() && npuendIndex >= npustartIndex) {
            int flag = 0;
            onnx::GraphProto lastSubgraph;
            for (int j = npustartIndex; j < subgraph.node_size(); ++j) {
                // 在 vector 中查找目标字符串
                auto it = std::find(NPUPreferOp.begin(), NPUPreferOp.end(), subgraph.node(j).op_type());
                if (it != NPUPreferOp.end()) {
                    flag = 1;
                }
                *lastSubgraph.add_node() = subgraph.node(j);
            }
            if (flag) {
                NPUSubgraphs.push_back(lastSubgraph);
            } else if (npuendIndex - npustartIndex + 1 > 10) { //TODO这里有待商榷
                NPUSubgraphs.push_back(lastSubgraph);
            } else {
                CPUSubgraphs.push_back(lastSubgraph);
            }
        } else if (cpustartIndex < subgraph.node_size() && cpuendIndex >= cpustartIndex) {
            onnx::GraphProto lastSubgraph;
            for (int j = cpustartIndex; j < subgraph.node_size(); ++j) {
                *lastSubgraph.add_node() = subgraph.node(j);
            }
            CPUSubgraphs.push_back(lastSubgraph);
        }
        
    }

    Subgraphs_ = NPUSubgraphs;
    return CPUSubgraphs;
}

std::vector<onnx::GraphProto> processCpuSubgraphs(std::vector<onnx::GraphProto>& Subgraphs_, const std::vector<std::string>& CPUSupportOp) {
    // 创建一个新的向量来存储包含 NPUSupportOp 的子图
    std::vector<onnx::GraphProto> NPUSubgraphs;
    std::vector<onnx::GraphProto> CPUSubgraphs;

    // 遍历每个子图
    for (size_t i = 0; i < Subgraphs_.size(); ++i) {
        auto subgraph = Subgraphs_[i];

        // 遍历当前子图中的每个节点
        int npustartIndex = 0;
        int npuendIndex = -1;
        int cpustartIndex = 0;
        int cpuendIndex = -1;

        for (int j = 0; j < subgraph.node_size(); ++j) {
            const onnx::NodeProto& node = subgraph.node(j);

            // 检查当前节点的操作是否在 CPUSupportOp 中
            if (std::find(CPUSupportOp.begin(), CPUSupportOp.end(), node.op_type()) == CPUSupportOp.end()) {
                if (cpuendIndex >= cpustartIndex) {
                    onnx::GraphProto newSubgraph;
                    for (int k = cpustartIndex; k <= cpuendIndex; ++k) {
                        *newSubgraph.add_node() = subgraph.node(k);
                    }
                    CPUSubgraphs.push_back(newSubgraph);
                }
                npuendIndex = j;
                cpustartIndex = j + 1;
            } else {
                if (npuendIndex >= npustartIndex) {
                    onnx::GraphProto newSubgraph;
                    for (int k = npustartIndex; k <= npuendIndex; ++k) {
                        *newSubgraph.add_node() = subgraph.node(k);
                    }
                    NPUSubgraphs.push_back(newSubgraph);
                }
                cpuendIndex = j;
                npustartIndex = j + 1;
            }
        }

        if (npustartIndex < subgraph.node_size() && npuendIndex >= npustartIndex) {
            onnx::GraphProto lastSubgraph;
            for (int j = npustartIndex; j < subgraph.node_size(); ++j) {
                *lastSubgraph.add_node() = subgraph.node(j);
            }
            NPUSubgraphs.push_back(lastSubgraph);
        } else if (cpustartIndex < subgraph.node_size() && cpuendIndex >= cpustartIndex) {
            onnx::GraphProto lastSubgraph;
            for (int j = cpustartIndex; j < subgraph.node_size(); ++j) {
                *lastSubgraph.add_node() = subgraph.node(j);
            }
            CPUSubgraphs.push_back(lastSubgraph);
        }
        
    }

    Subgraphs_ = CPUSubgraphs;
    return CPUSubgraphs;
}

std::vector<onnx::GraphProto> CheckWhetherNpuSupports(std::vector<onnx::GraphProto>& Subgraphs_, Device& d) {
    return processNpuSubgraphs(Subgraphs_, d.getNPUSupportOp(), d.getNPUPreferOp());
}

std::vector<onnx::GraphProto> CheckWhetherCpuSupports(std::vector<onnx::GraphProto>& Subgraphs_, Device& d) {
    return processCpuSubgraphs(Subgraphs_, d.getCPUSupportOp());
}

void Partition::PartitionGraph(const onnx::GraphProto &g, Device& d, PartitionStrategy strategy, const std::unordered_map<std::string, NodeIOSize> &node_io_size) {
    //std::unordered_set<NodeTensor> initializerNames = getInitializer(g);
    std::unordered_set<NodeTensor> IOvalueNames = getIOvalue(g);
    int* visited = (int*)malloc(g.node_size()*sizeof(int));
    std::vector<graph_adjacency_node> adjacency_list=get_adjancency_list(g, visited);
    std::vector<onnx::GraphProto> otherSubgraphs;
    // int temp_index=0;
    // for(auto& node : adjacency_list)
    // {
    //     std::cout<<"adjacency "<<temp_index<<" name:"<<g.node(temp_index).name()<<" ";
    //     for(int i=0;i<node.output_node_index.size();i++)
    //     {
    //         std::cout<<node.output_node_index[i]<<" ";
    //     }
    //     std::cout<<std::endl;
    //     temp_index++;
    // }
    determine_subgraphs(g,otherSubgraphs, d, visited, adjacency_list,strategy);
    free(visited);
    std::vector<graph_adjacency_node>().swap(adjacency_list);
    //findAndPrintStructures(g, d, strategy);////匹配主要平台(cpu/NPU)的结构并创立子图
    // //////////////////7.19
    // std::vector<std::unordered_set<std::string>> subgraphs_1_input_nodes_;
    // std::vector<std::unordered_set<std::string>> subgraphs_1_nodes_;
    // for (const auto& sg : Subgraphs) {
    //     std::unordered_set<NodeTensor> graphInputs;
    //     determineGraphInput(sg, IOvalueNames, graphInputs);////确定Subgraph中的每个子图的输入量

    //     // 根据输入
    //     std::unordered_set<std::string> graphInputsNodes;
    //     for (const auto& input : graphInputs) {
    //         auto nodename = findInputNode(g, input.name);////找到每个子图输入对应的网络节点输出
    //         if (nodename != "") {
    //             graphInputsNodes.insert(nodename);////所有subgraph的输入节点
    //         }
    //     }
    //     subgraphs_1_input_nodes_.push_back(graphInputsNodes);
    //     subgraphs_1_nodes_.push_back(collectNodeNames(sg));////所有subgraph的所有节点
    // }

    // for (size_t i = 0; i < Subgraphs.size(); ++i) {
    //     if (subgraphs_1_input_nodes_[i].empty()) {
    //         int mergeIndex = canMerge(i, subgraphs_1_input_nodes_, subgraphs_1_nodes_[i]);////第mergeIndex个子图的输入节点全在subgraphs_1_nodes_[i]里，可以被合并进这个子图
    //         if (mergeIndex != -1) {
    //             std::cout << "Merge possible for graphs " << i << " and " << mergeIndex << std::endl;
    //             // Merge the graphs
    //             if (i < mergeIndex) {
    //                 mergeGraphs(Subgraphs[i], Subgraphs[mergeIndex]);
    //                 Subgraphs.erase(Subgraphs.begin() + mergeIndex);
    //             } else {
    //                 mergeGraphs(Subgraphs[mergeIndex], Subgraphs[i]);
    //                 Subgraphs.erase(Subgraphs.begin() + i);
    //             }

    //             if (mergeIndex < i) {
    //                 i--;
    //             }
    //         }
    //     }
    // }
    // ////////////////////end
    ///////10.12注释
    int node_sum = 0;
    // 遍历结构并打印每个元素
    std::ofstream outFile("./subgraphs_1.txt");
    if (!outFile.is_open()) {
        std::cerr << "Error opening file." << std::endl;
        exit(0);
    }
    int id = 0;
    for (const auto& vec : Subgraphs) {
        outFile << " subgraph" << id << ":";
        for (const auto& node : vec.node()) {
            outFile << node.name() << " ";
        }
        id++;
        outFile << std::endl;
        node_sum += vec.node_size();
    }
    int id_record = id;
    //auto otherSubgraphs = findOtherSubgraphs(g, Subgraphs);//////////////////////////////////
    std::ofstream outFile_2("./subgraphs_2.txt");
    if (!outFile_2.is_open()) {
        std::cerr << "Error opening file." << std::endl;
        exit(0);
    }
    std::cout << "before:" << std::endl;
    for (const auto& vec : otherSubgraphs) {
        outFile_2 << " subgraph" << id << ":";
        for (const auto& node : vec.node()) {
            outFile_2 << node.name() << " ";
        }
        id++;
        outFile_2 << std::endl;
        node_sum += vec.node_size();
    }////把未知子图对应的node存入文件

    // 由于大型网络顺序遍历其node，往往并不是按照bfs的拓扑来的，因此切出来的图可能有些是可以合并的，因此这里要检查一下
    // 获得所有同类型子图的输入tensor的节点名称，同时统计每个子图的node名称，若某个子图的所有输入节点都来自某一个子图，认为这俩可以合并

//两子图合并，若后一个子图有来自于其他设备的子图，则 index+1,合并后的子图，不能大于1！

    std::vector<std::unordered_set<std::string>> subgraphs_2_input_nodes_;
    std::vector<std::unordered_set<std::string>> subgraphs_2_nodes_;
    for (const auto& sg : otherSubgraphs) {
        std::unordered_set<NodeTensor> graphInputs;
        determineGraphInput(sg, IOvalueNames, graphInputs);////确定otherSubgraph中的每个子图的输入量

        // 根据输入
        std::unordered_set<std::string> graphInputsNodes;
        for (const auto& input : graphInputs) {
            auto nodename = findInputNode(g, input.name);////找到每个子图输入对应的网络节点输出
            if (nodename != "") {
                graphInputsNodes.insert(nodename);////所有othersubgraph的输入节点
            }
        }
        subgraphs_2_input_nodes_.push_back(graphInputsNodes);
        subgraphs_2_nodes_.push_back(collectNodeNames(sg));////所有othersubgraph的所有节点
    }
//逻辑有问题，merge要重写
//merge完之后把需要删掉的图做个标记（在图上做），最后统一删除，或者从后往前删除
    int* is_merged = (int *)malloc(otherSubgraphs.size() * sizeof(int));
    for(int i=0;i<otherSubgraphs.size();i++)
    {
        is_merged[i] = 0;
    }
    for (size_t i = 0; i < otherSubgraphs.size(); ++i) 
    {
        if(is_merged[i] == 0)
        {
            for (size_t j = i + 1; j < otherSubgraphs.size(); j++) {
                if(is_merged[j] == 0)
                {
                    for (const auto& InputnodeName : subgraphs_2_input_nodes_[j]) 
                    {
                        if (subgraphs_2_nodes_[i].find(InputnodeName) != subgraphs_2_nodes_[i].end()) 
                        {   
                            std::cout << "Merge possible for graphs " << i << " and " << j << std::endl;
                            mergeGraphs(otherSubgraphs[i], otherSubgraphs[j]);
                            is_merged[j] = 1;
                            break;
                        }
                    }
                    if(is_merged[j] == 0)
                    {
                        for (const auto& InputnodeName : subgraphs_2_input_nodes_[i]) 
                        {
                            if (subgraphs_2_nodes_[j].find(InputnodeName) != subgraphs_2_nodes_[j].end()) 
                            {   
                                std::cout << "Merge possible for graphs " << i << " and " << j << std::endl;
                                mergeGraphs(otherSubgraphs[i], otherSubgraphs[j]);
                                is_merged[j] = 1;
                                break;
                            }
                        }                    
                    }
                }
            }
        }
    }
    for(int i = otherSubgraphs.size()-1; i>=0; i--)
    {
        if(is_merged[i] == 1)
        {
            otherSubgraphs.erase(otherSubgraphs.begin() + i);
        }
    }
    std::cout << "graph size after merging:"<< otherSubgraphs.size() << std::endl;
//////////////////////
    // for (size_t i = 0; i < otherSubgraphs.size(); ++i) {
    //     if (subgraphs_2_input_nodes_[i].empty()) {
    //         int mergeIndex = canMerge(i, subgraphs_2_input_nodes_, subgraphs_2_nodes_[i]);////第mergeIndex个子图的输入节点全在subgraphs_2_nodes_[i]里，可以被合并进这个子图
    //         if (mergeIndex != -1) {
    //             std::cout << "Merge possible for graphs " << i << " and " << mergeIndex << std::endl;
    //             // Merge the graphs
    //             if (i < mergeIndex) {
    //                 mergeGraphs(otherSubgraphs[i], otherSubgraphs[mergeIndex]);
    //                 //otherSubgraphs.erase(otherSubgraphs.begin() + mergeIndex);
    //                 is_merged[mergeIndex] = 1;
    //             } else {
    //                 mergeGraphs(otherSubgraphs[mergeIndex], otherSubgraphs[i]);
    //                 //otherSubgraphs.erase(otherSubgraphs.begin() + i);
    //                 is_merged[i] = 1;
    //             }
    //             if (mergeIndex < i) {
    //                 i--;
    //             }
    //         }
    //     }
    // }
    //////////////////////////////
    free(is_merged);
    std::ofstream outFile_3("./subgraphs_3.txt");
    if (!outFile_3.is_open()) {
        std::cerr << "Error opening file." << std::endl;
        exit(0);
    }
////merge后的othersubgraph
    for (const auto& vec : otherSubgraphs) {
        outFile_3 << " subgraph" << id_record << ":";
        for (const auto& node : vec.node()) {
            outFile_3 << node.name() << " ";
        }
        id_record++;
        outFile_3 << std::endl;
    }

    std::cout << "graph node size:" << g.node_size() << std::endl;
    std::cout << "sub node size:" << node_sum << std::endl;


    std::vector<std::unordered_set<NodeTensor>> subgraphs_1_inputs;
    std::vector<std::unordered_set<std::string>> subgraphs_1_input_nodes;
    std::vector<std::unordered_set<std::string>> subgraphs_1_nodes;
    for (const auto& sg : Subgraphs) {
        std::unordered_set<NodeTensor> graphInputs;
        determineGraphInput(sg, IOvalueNames, graphInputs);
        // if (graphInputs.empty() == true)
        // std::cout << "graphInputs are empty" << std::endl;
        subgraphs_1_inputs.push_back(graphInputs);

        // 根据输入
        std::unordered_set<std::string> graphInputsNodes;
        for (const auto& input : graphInputs) {
            auto nodename = findInputNode(g, input.name);
            if (nodename != "") {
                graphInputsNodes.insert(nodename);
            }
        }
        subgraphs_1_input_nodes.push_back(graphInputsNodes);
        subgraphs_1_nodes.push_back(collectNodeNames(sg));
    }

    std::vector<std::unordered_set<NodeTensor>> subgraphs_2_inputs;
    std::vector<std::unordered_set<std::string>> subgraphs_2_input_nodes;
    std::vector<std::unordered_set<std::string>> subgraphs_2_nodes;
    for (const auto& sg : otherSubgraphs) {
        std::unordered_set<NodeTensor> graphInputs;
        determineGraphInput(sg, IOvalueNames, graphInputs);
        subgraphs_2_inputs.push_back(graphInputs);

        // 根据输入
        std::unordered_set<std::string> graphInputsNodes;
        for (const auto& input : graphInputs) {
            auto nodename = findInputNode(g, input.name);
            if (nodename != "") {
                graphInputsNodes.insert(nodename);
            }
        }
        subgraphs_2_input_nodes.push_back(graphInputsNodes);
        subgraphs_2_nodes.push_back(collectNodeNames(sg));
    }
    ////merge后的othersubgraph的输入node和所有node
    //得出了全部的input后再确定output
    std::vector<std::unordered_set<NodeTensor>> subgraphs_1_outputs;

    int node_number=0;

    for (const auto& sg : Subgraphs) {
        std::unordered_set<NodeTensor> graphOutputs;
        node_number+=sg.node_size();
        determineGraphOutput(g,sg, subgraphs_1_inputs, subgraphs_2_inputs, graphOutputs);
        subgraphs_1_outputs.push_back(graphOutputs);
    }
    std::vector<std::unordered_set<NodeTensor>> subgraphs_2_outputs;
    for (const auto& sg : otherSubgraphs) {
        std::unordered_set<NodeTensor> graphOutputs;
        node_number+=sg.node_size();
        determineGraphOutput(g,sg, subgraphs_1_inputs, subgraphs_2_inputs, graphOutputs);
        subgraphs_2_outputs.push_back(graphOutputs);
    }
    
    std::cout<<"total number of nodes in subgraphs:"<<node_number<<std::endl;
    std::cout<<"total number of nodes in origional graph:"<<g.node_size()<<std::endl;

/////////////////////////6.29
    //std::unordered_set<NodeTensor> graphs_outputs=getOutvalue(g);

    std::vector<std::unordered_set<NodeTensor>> graphs_inputs;
    graphs_inputs.insert(graphs_inputs.end(),subgraphs_1_inputs.begin(),subgraphs_1_inputs.end());
    //std::cout <<  " size1:"<<graphs_inputs.size()<<std::endl;
    graphs_inputs.insert(graphs_inputs.end(),subgraphs_2_inputs.begin(),subgraphs_2_inputs.end());
    //std::cout <<  " size2:"<<graphs_inputs.size()<<std::endl;
    std::vector<std::unordered_set<NodeTensor>> graphs_outputs;
    graphs_outputs.insert(graphs_outputs.end(),subgraphs_1_outputs.begin(),subgraphs_1_outputs.end());
    graphs_outputs.insert(graphs_outputs.end(),subgraphs_2_outputs.begin(),subgraphs_2_outputs.end());

    std::vector<std::vector<int>> predecessors_Subgraphs(graphs_inputs.size());
    std::vector<std::vector<int>> successors_Subgraphs(graphs_inputs.size());
    ////可以在此处插入强连通分量处理相关////////////////////////////////////////////
    //把predecessors和successors计算放在前面
    for(int i=0; i<graphs_inputs.size();i++) ////遍历所有子图
    {
        std::vector<int> predecessors;
        for(const auto& g_input : graphs_inputs[i])////遍历某个子图的所有input
        {
            for(int j=0; j< graphs_outputs.size();j++)////检查该子图的某个input是否是第j个子图的output
            {
                if((graphs_outputs[j].find(g_input)!=graphs_outputs[j].end()))////该子图的某个input是第j个子图的output
                {
                    predecessors.push_back(j);//加入前驱
                }
            }
        }
        predecessors_Subgraphs[i].insert(predecessors_Subgraphs[i].end(),predecessors.begin(),predecessors.end());
    }
    for(int i=0;i<graphs_inputs.size();i++)
    {
        for(int j=0;j<graphs_inputs.size();j++)
        {
            if(find(predecessors_Subgraphs[j].begin(),predecessors_Subgraphs[j].end(),i)!=predecessors_Subgraphs[j].end())
            {
                successors_Subgraphs[i].push_back(j);
            }
        }
    }
    std::vector<std::vector<int>> strongly_connected_subgraphs;
    int* DFN = (int *)malloc(graphs_inputs.size() * sizeof(int));
    int* LOW = (int *)malloc(graphs_inputs.size() * sizeof(int));
    for(int i = 0; i < graphs_inputs.size(); i++)
    {
        DFN[i] = 0;
        LOW[i] = 0;
    }
    int temp_count = 0;
    for(const auto& predecessors : predecessors_Subgraphs)
    {
        if(DFN[temp_count] == 0)
        {
            std::vector<int> stack_subgraphs;
            int depth = 0;
            Tarjan(temp_count, depth, strongly_connected_subgraphs, DFN, 
            LOW, stack_subgraphs, successors_Subgraphs);
        }
        temp_count ++ ;
    } 
    
    std::string file_name_scc = "scc.txt";
    std::ofstream outfile_scc(file_name_scc);    
    for(const auto& scc : strongly_connected_subgraphs)
    {
        std::cout << "scc:";
        for(const auto& scc_id : scc)
        {
            std::cout << scc_id << " ";
            outfile_scc << "subgraph" << scc_id << " input:";
            for(const auto& scc_input : graphs_inputs[scc_id])
            {
                outfile_scc << scc_input.name << ";";
            }
            outfile_scc << " output:";
            for(const auto& scc_output : graphs_outputs[scc_id])
            {
                outfile_scc << scc_output.name << ";";
            }
            outfile_scc << std::endl;
        }
        
        std::cout << std::endl;
    }
    outfile_scc.close();
    ////////////////////
    ////////////////
    //std::vector<std::vector<std::vector<int>>> sccs_pairs;
    // find_subgraph_pair(strongly_connected_subgraphs, Subgraphs,otherSubgraphs,graphs_inputs,graphs_outputs,sccs_pairs);
    // for(const auto& scc_pairs : sccs_pairs)
    // {
    //     for(const auto& scc_pair : scc_pairs)
    //     {
    //         std::cout << "scc pair:";
    //         for(const auto& scc_id : scc_pair)
    //         {
    //             std::cout << scc_id << " ";
    //         }
    //         std::cout << std::endl;
    //     }
    // }

    // for(auto scc_pairs : sccs_pairs)
    // {
    //     for(auto scc_pair : scc_pairs)
    //     {
    //         std::vector<onnx::GraphProto> scc_pair_cut;
    //         std::vector<int> cut_info = cut_pair(Subgraphs, otherSubgraphs, graphs_inputs, graphs_outputs, scc_pair, scc_pair_cut);
    //         std::cout << "cut info:";
    //         for(const auto& info : cut_info)
    //         {
    //             std::cout << info << " ";
                
    //         }
    //         std::cout << "scc pair cut size:"<< scc_pair_cut[0].node_size() << "+ " << scc_pair_cut[1].node_size();
    //         std::cout << std::endl;
    //     }
    // }
    
    free(DFN);
    free(LOW);
    // std::string file_name_predecessor = "predecessor.txt";
    // std::string file_name_successor = "successor.txt";
    // std::ofstream outfile_predecessor(file_name_predecessor);
    // std::ofstream outfile_successor(file_name_successor);
    // if (!(outfile_predecessor.is_open()&&outfile_successor.is_open())) {
    //     std::cerr << "Error opening file." << std::endl;
    //     exit(0);
    // }
    // for(int i=0;i<graphs_inputs.size();i++)
    // {
    //     outfile_predecessor << "predecessor of subgraph " << i << ":";
    //     for(const auto& predecessor : predecessors_Subgraphs[i])
    //     {
    //         outfile_predecessor << predecessor << ";";
    //     }
    //     outfile_predecessor << std::endl;
    //     outfile_successor << "successor of subgraph " << i << ":";
    //     for(const auto& successor : successors_Subgraphs[i])
    //     {
    //         outfile_successor << successor << ";";
    //     }
    //     outfile_successor << std::endl;
    // }
    // outfile_predecessor.close();
    // outfile_successor.close();
    
    //eliminate_connection(strongly_connected_subgraphs, Subgraphs, otherSubgraphs, graphs_outputs);

    eliminate_pair(Subgraphs, otherSubgraphs, graphs_inputs, graphs_outputs, strongly_connected_subgraphs);
    //std::exit(0);
    strongly_connected_subgraphs.clear();
    predecessors_Subgraphs.clear();
    successors_Subgraphs.clear();
    subgraphs_1_inputs.clear();
    std::vector<std::unordered_set<std::string>>().swap(subgraphs_1_input_nodes);
    std::vector<std::unordered_set<std::string>>().swap(subgraphs_1_nodes);
    subgraphs_2_inputs.clear();
    std::vector<std::unordered_set<std::string>>().swap(subgraphs_2_input_nodes);
    std::vector<std::unordered_set<std::string>>().swap(subgraphs_2_nodes);
    subgraphs_1_outputs.clear();
    subgraphs_2_outputs.clear();
    graphs_inputs.clear();
    graphs_outputs.clear();
    std::cout << "Tarjan ended!!!!!!!!!!!" << std::endl;
    for (const auto& sg : Subgraphs) {
        std::unordered_set<NodeTensor> graphInputs;
        determineGraphInput(sg, IOvalueNames, graphInputs);
        // if (graphInputs.empty() == true)
        // std::cout << "graphInputs are empty" << std::endl;
        subgraphs_1_inputs.push_back(graphInputs);

        // 根据输入
        // std::unordered_set<std::string> graphInputsNodes;
        // for (const auto& input : graphInputs) {
        //     auto nodename = findInputNode(g, input.name);
        //     if (nodename != "") {
        //         graphInputsNodes.insert(nodename);
        //     }
        // }
        //subgraphs_1_input_nodes.push_back(graphInputsNodes);
        //subgraphs_1_nodes.push_back(collectNodeNames(sg));
    }
    for (const auto& sg : otherSubgraphs) {
        std::unordered_set<NodeTensor> graphInputs;
        determineGraphInput(sg, IOvalueNames, graphInputs);
        subgraphs_2_inputs.push_back(graphInputs);
        // for(const auto& input : graphInputs)
        // {
        //     std::cout << input.name << " ";
        // }
        // std::cout << std::endl;

        // 根据输入
        // std::unordered_set<std::string> graphInputsNodes;
        // for (const auto& input : graphInputs) {
        //     auto nodename = findInputNode(g, input.name);
        //     if (nodename != "") {
        //         graphInputsNodes.insert(nodename);
        //     }
        // }
        //subgraphs_2_input_nodes.push_back(graphInputsNodes);
        //subgraphs_2_nodes.push_back(collectNodeNames(sg));
    }
    node_number = 0;
    for (const auto& sg : Subgraphs) {
        std::unordered_set<NodeTensor> graphOutputs;
        node_number+=sg.node_size();
        determineGraphOutput(g,sg, subgraphs_1_inputs, subgraphs_2_inputs, graphOutputs);
        subgraphs_1_outputs.push_back(graphOutputs);
    }
    for (const auto& sg : otherSubgraphs) {
        std::unordered_set<NodeTensor> graphOutputs;
        node_number+=sg.node_size();
        determineGraphOutput(g,sg, subgraphs_1_inputs, subgraphs_2_inputs, graphOutputs);
        subgraphs_2_outputs.push_back(graphOutputs);
    }
    //std::cout<<"no problem here"<<std::endl;
    graphs_inputs.insert(graphs_inputs.end(),subgraphs_1_inputs.begin(),subgraphs_1_inputs.end());
    graphs_inputs.insert(graphs_inputs.end(),subgraphs_2_inputs.begin(),subgraphs_2_inputs.end());
    graphs_outputs.insert(graphs_outputs.end(),subgraphs_1_outputs.begin(),subgraphs_1_outputs.end());
    graphs_outputs.insert(graphs_outputs.end(),subgraphs_2_outputs.begin(),subgraphs_2_outputs.end());
    for(int i=0; i<graphs_inputs.size();i++) ////遍历所有子图
    {
        std::vector<int> predecessors;
        for(const auto& g_input : graphs_inputs[i])////遍历某个子图的所有input
        {
            for(int j=0; j< graphs_outputs.size();j++)////检查该子图的某个input是否是第j个子图的output
            {
                if((graphs_outputs[j].find(g_input)!=graphs_outputs[j].end()))////该子图的某个input是第j个子图的output
                {
                    predecessors.push_back(j);//加入前驱
                }
            }
        }
        //predecessors_Subgraphs[i].insert(predecessors_Subgraphs[i].end(),predecessors.begin(),predecessors.end());
        predecessors_Subgraphs.push_back(predecessors);
    }
    for(int i=0;i<graphs_inputs.size();i++)
    {
        std::vector<int> temp;
        for(int j=0;j<graphs_inputs.size();j++)
        {
            if(find(predecessors_Subgraphs[j].begin(),predecessors_Subgraphs[j].end(),i)!=predecessors_Subgraphs[j].end())
            {
                temp.push_back(j);
            }
        }
        successors_Subgraphs.push_back(temp);
    }
    //需要将predecessors和succeedors输出到文件中
    std::string file_name_predecessor = "predecessor_final.txt";
    std::string file_name_successor = "successor_final.txt";
    std::ofstream outfile_predecessor(file_name_predecessor);
    std::ofstream outfile_successor(file_name_successor);
    if (!(outfile_predecessor.is_open()&&outfile_successor.is_open())) {
        std::cerr << "Error opening file." << std::endl;
        exit(0);
    }
    for(int i=0;i<graphs_inputs.size();i++)
    {
        outfile_predecessor << "predecessor of subgraph " << i << ":";
        for(const auto& predecessor : predecessors_Subgraphs[i])
        {
            outfile_predecessor << predecessor << ";";
        }
        outfile_predecessor << std::endl;
        outfile_successor << "successor of subgraph " << i << ":";
        for(const auto& successor : successors_Subgraphs[i])
        {
            outfile_successor << successor << ";";
        }
        outfile_successor << std::endl;
    }
    outfile_predecessor.close();
    outfile_successor.close();
    print_subgraphs(Subgraphs, "./subgraphs_final.txt", otherSubgraphs, "./other_subgraphs_final.txt");
    
    // for(const auto& predecessors : predecessors_Subgraphs)
    // {
    //     if(DFN[temp_count] == 0)
    //     {
    //         std::vector<int> stack_subgraphs;
    //         int depth = 0;
    //         Tarjan(temp_count, depth, strongly_connected_subgraphs, DFN, 
    //         LOW, stack_subgraphs, successors_Subgraphs);
    //     }
    //     temp_count ++ ;
    // } 
    // if(strongly_connected_subgraphs.size()!= 0)
    // {
    //     std::cout << "new strongly connected subgraphs found" << std::endl;
    //     std::string file_name_scc_new = "scc_new.txt";
    //     std::ofstream outfile_scc_new(file_name_scc_new);    
    //     for(const auto& scc : strongly_connected_subgraphs)
    //     {
    //         std::cout << "scc:";
    //         for(const auto& scc_id : scc)
    //         {
    //             std::cout << scc_id << " ";
    //             outfile_scc_new << "subgraph" << scc_id << " input:";
    //             for(const auto& scc_input : graphs_inputs[scc_id])
    //             {
    //                 outfile_scc_new << scc_input.name << ";";
    //             }
    //             outfile_scc_new << " output:";
    //             for(const auto& scc_output : graphs_outputs[scc_id])
    //             {
    //                 outfile_scc_new << scc_output.name << ";";
    //             }
    //             outfile_scc_new << std::endl;
    //         }
            
    //         std::cout << std::endl;
    //     }
    //     outfile_scc_new.close();
    //     std::exit(0);
    // }
    ////////////////////////////////////////////////////
    int node_num_all = 0;
    for(const auto& sg : Subgraphs)
    {
        node_num_all += sg.node_size();
    }
    for(const auto& sg : otherSubgraphs)
    {
        node_num_all += sg.node_size();
    }
    std::cout << "node num in original graph: " << g.node_size() << std::endl;
    std::cout << "node_num after cut " << node_num_all << std::endl;
    ///////////////////////+++
    int* DFN_ = (int *)malloc(graphs_inputs.size() * sizeof(int));
    int* LOW_ = (int *)malloc(graphs_inputs.size() * sizeof(int));
    for(int i = 0; i < graphs_inputs.size(); i++)
    {
        DFN_[i] = 0;
        LOW_[i] = 0;
    }
    temp_count = 0;
    for(const auto& predecessors : predecessors_Subgraphs)
    {
        if(DFN_[temp_count] == 0)
        {
            std::vector<int> stack_subgraphs;
            int depth = 0;
            Tarjan(temp_count, depth, strongly_connected_subgraphs, DFN_, 
            LOW_, stack_subgraphs, successors_Subgraphs);
        }
        temp_count ++ ;
    } 
    free(DFN_);
    free(LOW_);
    eliminate_scc(strongly_connected_subgraphs,  Subgraphs, otherSubgraphs);
    /////////////////////    
    strongly_connected_subgraphs.clear();
    predecessors_Subgraphs.clear();
    successors_Subgraphs.clear();
    std::vector<std::unordered_set<NodeTensor>>().swap(subgraphs_2_inputs);
    std::vector<std::unordered_set<NodeTensor>>().swap(subgraphs_1_inputs);
    std::vector<std::unordered_set<NodeTensor>>().swap(subgraphs_2_outputs);
    std::vector<std::unordered_set<NodeTensor>>().swap(subgraphs_1_outputs);
    std::vector<std::unordered_set<NodeTensor>>().swap(graphs_inputs);
    std::vector<std::unordered_set<NodeTensor>>().swap(graphs_outputs);
    std::cout << "Tarjan ended!!!!!!!!!!!" << std::endl;
    for (const auto& sg : Subgraphs) {
        std::unordered_set<NodeTensor> graphInputs;
        determineGraphInput(sg, IOvalueNames, graphInputs);
        subgraphs_1_inputs.push_back(graphInputs);
    }
    for (const auto& sg : otherSubgraphs) {
        std::unordered_set<NodeTensor> graphInputs;
        determineGraphInput(sg, IOvalueNames, graphInputs);
        subgraphs_2_inputs.push_back(graphInputs);
    }
    node_number = 0;
    for (const auto& sg : Subgraphs) {
        std::unordered_set<NodeTensor> graphOutputs;
        node_number+=sg.node_size();
        determineGraphOutput(g,sg, subgraphs_1_inputs, subgraphs_2_inputs, graphOutputs);
        subgraphs_1_outputs.push_back(graphOutputs);
    }
    for (const auto& sg : otherSubgraphs) {
        std::unordered_set<NodeTensor> graphOutputs;
        node_number+=sg.node_size();
        determineGraphOutput(g,sg, subgraphs_1_inputs, subgraphs_2_inputs, graphOutputs);
        subgraphs_2_outputs.push_back(graphOutputs);
    }
    //std::cout<<"no problem here"<<std::endl;
    graphs_inputs.insert(graphs_inputs.end(),subgraphs_1_inputs.begin(),subgraphs_1_inputs.end());
    graphs_inputs.insert(graphs_inputs.end(),subgraphs_2_inputs.begin(),subgraphs_2_inputs.end());
    graphs_outputs.insert(graphs_outputs.end(),subgraphs_1_outputs.begin(),subgraphs_1_outputs.end());
    graphs_outputs.insert(graphs_outputs.end(),subgraphs_2_outputs.begin(),subgraphs_2_outputs.end());
    for(int i=0; i<graphs_inputs.size();i++) ////遍历所有子图
    {
        std::vector<int> predecessors;
        for(const auto& g_input : graphs_inputs[i])////遍历某个子图的所有input
        {
            for(int j=0; j< graphs_outputs.size();j++)////检查该子图的某个input是否是第j个子图的output
            {
                if((graphs_outputs[j].find(g_input)!=graphs_outputs[j].end()))////该子图的某个input是第j个子图的output
                {
                    predecessors.push_back(j);//加入前驱
                }
            }
        }
        //predecessors_Subgraphs[i].insert(predecessors_Subgraphs[i].end(),predecessors.begin(),predecessors.end());
        predecessors_Subgraphs.push_back(predecessors);
    }
    for(int i=0;i<graphs_inputs.size();i++)
    {
        std::vector<int> temp;
        for(int j=0;j<graphs_inputs.size();j++)
        {
            if(find(predecessors_Subgraphs[j].begin(),predecessors_Subgraphs[j].end(),i)!=predecessors_Subgraphs[j].end())
            {
                temp.push_back(j);
            }
        }
        successors_Subgraphs.push_back(temp);
    }
    ////////////////////////////////////////////////
    std::string file_name_predecessor_2 = "predecessor_final_2.txt";
    std::string file_name_successor_2 = "successor_final_2.txt";
    std::ofstream outfile_predecessor_2(file_name_predecessor_2);
    std::ofstream outfile_successor_2(file_name_successor_2);
    if (!(outfile_predecessor_2.is_open()&&outfile_successor_2.is_open())) {
        std::cerr << "Error opening file." << std::endl;
        exit(0);
    }
    for(int i=0;i<graphs_inputs.size();i++)
    {
        outfile_predecessor_2 << "predecessor of subgraph " << i << ":";
        for(const auto& predecessor : predecessors_Subgraphs[i])
        {
            outfile_predecessor_2 << predecessor << ";";
        }
        outfile_predecessor_2 << std::endl;
        outfile_successor_2 << "successor of subgraph " << i << ":";
        for(const auto& successor : successors_Subgraphs[i])
        {
            outfile_successor_2 << successor << ";";
        }
        outfile_successor_2 << std::endl;
    }
    outfile_predecessor_2.close();
    outfile_successor_2.close();
    print_subgraphs(Subgraphs, "./subgraphs_final_2.txt", otherSubgraphs, "./other_subgraphs_final_2.txt");
    //////////////////////////////////////////-------
    ///////////////////////+++
    int* DFN_2 = (int *)malloc(graphs_inputs.size() * sizeof(int));
    int* LOW_2 = (int *)malloc(graphs_inputs.size() * sizeof(int));
    for(int i = 0; i < graphs_inputs.size(); i++)
    {
        DFN_2[i] = 0;
        LOW_2[i] = 0;
    }
    temp_count = 0;
    for(const auto& predecessors : predecessors_Subgraphs)
    {
        if(DFN_[temp_count] == 0)
        {
            std::vector<int> stack_subgraphs;
            int depth = 0;
            Tarjan(temp_count, depth, strongly_connected_subgraphs, DFN_2, 
            LOW_2, stack_subgraphs, successors_Subgraphs);
        }
        temp_count ++ ;
    } 
    std::ofstream outfile_scc2(file_name_scc);    
    for(const auto& scc : strongly_connected_subgraphs)
    {
        std::cout << "scc:";
        for(const auto& scc_id : scc)
        {
            std::cout << scc_id << " ";
            outfile_scc2 << "subgraph" << scc_id << " input:";
            for(const auto& scc_input : graphs_inputs[scc_id])
            {
                outfile_scc2 << scc_input.name << ";";
            }
            outfile_scc2 << " output:";
            for(const auto& scc_output : graphs_outputs[scc_id])
            {
                outfile_scc2 << scc_output.name << ";";
            }
            outfile_scc2 << std::endl;
        }
        
        std::cout << std::endl;
    }
    outfile_scc.close();
    free(DFN_2);
    free(LOW_2);
    eliminate_scc_v2(strongly_connected_subgraphs,  Subgraphs, otherSubgraphs, g);
    /////////clear   
    strongly_connected_subgraphs.clear();
    predecessors_Subgraphs.clear();
    successors_Subgraphs.clear();
    std::vector<std::unordered_set<NodeTensor>>().swap(subgraphs_2_inputs);
    std::vector<std::unordered_set<NodeTensor>>().swap(subgraphs_1_inputs);
    std::vector<std::unordered_set<NodeTensor>>().swap(subgraphs_2_outputs);
    std::vector<std::unordered_set<NodeTensor>>().swap(subgraphs_1_outputs);
    std::vector<std::unordered_set<NodeTensor>>().swap(graphs_inputs);
    std::vector<std::unordered_set<NodeTensor>>().swap(graphs_outputs);
    std::cout <<"line:"<< __LINE__ << std::endl;
    for (const auto& sg : Subgraphs) {
        std::unordered_set<NodeTensor> graphInputs;
        determineGraphInput(sg, IOvalueNames, graphInputs);
        subgraphs_1_inputs.push_back(graphInputs);
    }
    std::cout <<"line:"<< __LINE__ << std::endl;
    for (const auto& sg : otherSubgraphs) {
        std::unordered_set<NodeTensor> graphInputs;
        determineGraphInput(sg, IOvalueNames, graphInputs);
        subgraphs_2_inputs.push_back(graphInputs);
    }
    std::cout <<"line:"<< __LINE__ << std::endl;
    node_number = 0;
    for (const auto& sg : Subgraphs) {
        std::unordered_set<NodeTensor> graphOutputs;
        node_number+=sg.node_size();
        determineGraphOutput(g,sg, subgraphs_1_inputs, subgraphs_2_inputs, graphOutputs);
        subgraphs_1_outputs.push_back(graphOutputs);
    }
    std::cout <<"line:"<< __LINE__ << std::endl;
    for (const auto& sg : otherSubgraphs) {
        std::unordered_set<NodeTensor> graphOutputs;
        node_number+=sg.node_size();
        determineGraphOutput(g,sg, subgraphs_1_inputs, subgraphs_2_inputs, graphOutputs);
        subgraphs_2_outputs.push_back(graphOutputs);
    }
    std::cout <<"line:"<< __LINE__ << std::endl;
    graphs_inputs.insert(graphs_inputs.end(),subgraphs_1_inputs.begin(),subgraphs_1_inputs.end());
    graphs_inputs.insert(graphs_inputs.end(),subgraphs_2_inputs.begin(),subgraphs_2_inputs.end());
    graphs_outputs.insert(graphs_outputs.end(),subgraphs_1_outputs.begin(),subgraphs_1_outputs.end());
    graphs_outputs.insert(graphs_outputs.end(),subgraphs_2_outputs.begin(),subgraphs_2_outputs.end());
    for(int i=0; i<graphs_inputs.size();i++) ////遍历所有子图
    {
        std::vector<int> predecessors;
        for(const auto& g_input : graphs_inputs[i])////遍历某个子图的所有input
        {
            for(int j=0; j< graphs_outputs.size();j++)////检查该子图的某个input是否是第j个子图的output
            {
                if((graphs_outputs[j].find(g_input)!=graphs_outputs[j].end()))////该子图的某个input是第j个子图的output
                {
                    predecessors.push_back(j);//加入前驱
                }
            }
        }
        //predecessors_Subgraphs[i].insert(predecessors_Subgraphs[i].end(),predecessors.begin(),predecessors.end());
        predecessors_Subgraphs.push_back(predecessors);
    }
    for(int i=0;i<graphs_inputs.size();i++)
    {
        std::vector<int> temp;
        for(int j=0;j<graphs_inputs.size();j++)
        {
            if(find(predecessors_Subgraphs[j].begin(),predecessors_Subgraphs[j].end(),i)!=predecessors_Subgraphs[j].end())
            {
                temp.push_back(j);
            }
        }
        successors_Subgraphs.push_back(temp);
    }
    //////file out predecessor and successor
    std::string file_name_predecessor_3 = "predecessor_final_3.txt";
    std::string file_name_successor_3 = "successor_final_3.txt";
    std::ofstream outfile_predecessor_3(file_name_predecessor_3);
    std::ofstream outfile_successor_3(file_name_successor_3);
    if (!(outfile_predecessor_3.is_open()&&outfile_successor_3.is_open())) {
        std::cerr << "Error opening file." << std::endl;
        exit(0);
    }
    for(int i=0;i<graphs_inputs.size();i++)
    {
        outfile_predecessor_3 << "predecessor of subgraph " << i << ":";
        for(const auto& predecessor : predecessors_Subgraphs[i])
        {
            outfile_predecessor_3 << predecessor << ";";
        }
        outfile_predecessor_3 << std::endl;
        outfile_successor_3 << "successor of subgraph " << i << ":";
        for(const auto& successor : successors_Subgraphs[i])
        {
            outfile_successor_3 << successor << ";";
        }
        outfile_successor_3 << std::endl;
    }
    outfile_predecessor_3.close();
    outfile_successor_3.close();
    print_subgraphs(Subgraphs, "./subgraphs_final_3.txt", otherSubgraphs, "./other_subgraphs_final_3.txt");
    node_num_all = 0;
    for(const auto& sg : Subgraphs)
    {
        node_num_all += sg.node_size();
    }
    for(const auto& sg : otherSubgraphs)
    {
        node_num_all += sg.node_size();
    }
    int* DFN_3 = (int *)malloc(graphs_inputs.size() * sizeof(int));
    int* LOW_3 = (int *)malloc(graphs_inputs.size() * sizeof(int));
    for(int i = 0; i < graphs_inputs.size(); i++)
    {
        DFN_3[i] = 0;
        LOW_3[i] = 0;
    }
    temp_count = 0;
    for(const auto& predecessors : predecessors_Subgraphs)
    {
        if(DFN_[temp_count] == 0)
        {
            std::vector<int> stack_subgraphs;
            int depth = 0;
            Tarjan(temp_count, depth, strongly_connected_subgraphs, DFN_3, 
            LOW_3, stack_subgraphs, successors_Subgraphs);
        }
        temp_count ++ ;
    } 
    std::string file_name_scc3 = "scc3.txt";
    std::ofstream outfile_scc3(file_name_scc3);    
    for(const auto& scc : strongly_connected_subgraphs)
    {
        std::cout << "scc:";
        for(const auto& scc_id : scc)
        {
            std::cout << scc_id << " ";
            outfile_scc3 << "subgraph" << scc_id << " input:";
            for(const auto& scc_input : graphs_inputs[scc_id])
            {
                outfile_scc3 << scc_input.name << ";";
            }
            outfile_scc3 << " output:";
            for(const auto& scc_output : graphs_outputs[scc_id])
            {
                outfile_scc3 << scc_output.name << ";";
            }
            outfile_scc3 << std::endl;
        }
        
        std::cout << std::endl;
    }
    outfile_scc.close();
    free(DFN_3);
    free(LOW_3);
    std::cout << "node num in original graph: " << g.node_size() << std::endl;
    std::cout << "node_num after cut " << node_num_all << std::endl;
    ////////////////////////////////////////////////
    std::cout << "succeeded in reaching sorting" << std::endl;
    int finished_flag=0;int sort_count=0;
    std::vector<int> order_Subgraphs(graphs_inputs.size());
    std::vector<int> issort_Subgraphs(graphs_inputs.size());
    while(!finished_flag) 
    {
        finished_flag=1;
        int changed_sort_flag=0;
        if(sort_count==0)
        {
            changed_sort_flag=1;
            for(int i=0; i<graphs_inputs.size();i++) ////遍历所有子图
            //for(int i = graphs_inputs.size()-1; i>=0;i--)
            {

                int find_flag=0;
                //std::cout << "no problem here-1<<<"<<i<<std::endl;
                for(const auto& g_input : graphs_inputs[i])////遍历某个子图的所有input
                {
                    #pragma unroll
                    for(int j=0; j< graphs_outputs.size();j++)////检查该子图的某个input是否是output
                    {
                        if(graphs_outputs[j].find(g_input)!=graphs_outputs[j].end())
                        {
                        find_flag=1;
                        break;
                        }
                    }
                    if(find_flag){break;}
                }
                if(!find_flag)
                {
                    order_Subgraphs[i]=0;
                    issort_Subgraphs[i]=1;
                }
                else {order_Subgraphs[i]=1;issort_Subgraphs[i]=0;finished_flag=0;}
                //std::cout << "no problem here2<<<"<<i<<std::endl;
            }
            //std::cout << "no problem here-1<<<"<<std::endl;
        }
        else
        {
            std::cout << "sort count:" <<sort_count << std::endl;
            for(int i=0; i<graphs_inputs.size();i++) ////遍历所有子图
            {
                int find_flag=0;
                //std::vector<int> predecessors;
                //std::cout << "no problem here0<<<"<<std::endl;
                if(issort_Subgraphs[i]==1&&i!=graphs_inputs.size()-1){continue;}////如果已经排过序了，跳过这个子图
                //std::cout << "no problem here1<<<"<<std::endl;
                for(const auto& g_input : graphs_inputs[i])////遍历某个子图的所有input
                {
                    for(int j=0; j< graphs_outputs.size();j++)////检查该子图的某个input是否是第j个子图的output
                    {
                        if((graphs_outputs[j].find(g_input)!=graphs_outputs[j].end()))////该子图的某个input是第j个子图的output
                        {
                            if((issort_Subgraphs[j]==0))//若第j个子图尚未被排序
                            {
                                std::cout << "graph "<<i << "is after graph "<<j << std::endl;
                                find_flag=1;
                                break;
                            }
                        }
                    }
                    if(find_flag){break;}
                }
                if(!find_flag)
                {
                    if(!issort_Subgraphs[i]==1)
                    {
                        order_Subgraphs[i]=sort_count;
                    }
                    
                    //changed_sort_flag = 1;
                    //predecessors_Subgraphs[i].insert(predecessors_Subgraphs[i].end(),predecessors.begin(),predecessors.end());
                }
                else {order_Subgraphs[i]=sort_count+1;issort_Subgraphs[i]=0;finished_flag=0;}
                if(i==graphs_inputs.size()-1)//本循环到最后再统一加入队列，防止出现本轮循环新加入的为后面子图的前驱
                {
                    for(int j=0; j<graphs_inputs.size();j++)
                    {
                        // if(issort_Subgraphs[j]==1&&j==graphs_inputs.size()-1&&order_Subgraphs[j]<sort_count)
                        // {
                        //     break;
                        // }
                        if(order_Subgraphs[j]==sort_count)
                        {
                            issort_Subgraphs[j]=1;
                            changed_sort_flag = 1;
                            std::cout << "graph "<<j << " is in the "<<sort_count << "th sort" << std::endl;
                        }
                    }
                }
            }
            if(changed_sort_flag == 0)
            {
                std::cout << "error: endless loop!" << std::endl;
                std::cout << "sort count:" <<sort_count << std::endl;
                for(int i =0;i<graphs_inputs.size();i++)
                {
                    std::cout << "order_Subgraphs["<<i<<"]:"<<order_Subgraphs[i]<<" ";
                }
                std::cout << std::endl;
                std::exit(1);
                break;
            }
        }
        sort_count++;
    }
    //find successors
    // for(int i=0;i<graphs_inputs.size();i++)
    // {
    //     for(int j=0;j<graphs_inputs.size();j++)
    //     {
    //         if(find(predecessors_Subgraphs[j].begin(),predecessors_Subgraphs[j].end(),i)!=predecessors_Subgraphs[j].end())
    //         {
    //             successors_Subgraphs[i].push_back(j);
    //         }
    //     }
    // }
    char* sub1_type,*sub2_type;
    if(strategy==SPILTE_CPU_STRUCTURE_FIRST)
    {
        sub1_type="CPU";
        sub2_type="NPU";
    }
    else{
        sub1_type="NPU";
        sub2_type="CPU";
    }
    std::cout <<  " order"<<std::endl;
    for(auto element : order_Subgraphs)
    {
        std::cout << element << " ";
    }
    std::cout<<std::endl;

    std::string file_name = "subgraphs_ios.txt";
    std::ofstream outfile1(file_name);
    if (!outfile1.is_open()) {
        std::cerr << "Error opening file." << std::endl;
        exit(0);
    }
    int sub1_size=subgraphs_1_inputs.size();
    int sub2_size=subgraphs_2_inputs.size();
    for(int i=0;i<graphs_inputs.size();i++)
    {
        outfile1 << (i>=sub1_size?sub2_type:sub1_type)<<"subgraph"<<(i>=sub1_size?(i-sub1_size):i)<<": order"<<order_Subgraphs[i];
        outfile1<<"--input-name ";
        std::cout << (i>=sub1_size?sub2_type:sub1_type)<<"subgraph"<<(i>=sub1_size?(i-sub1_size):i)<<": order"<<order_Subgraphs[i]<<std::endl;
        std::cout << "Inputs:";
        for(auto element :  graphs_inputs[i])
        {
            std::cout<<element.name<<"; size:";
            for(auto Size : element.shape)
            {std::cout<<Size<<" ";}
            outfile1<<element.name<<";";
        }
        std::cout << std::endl;
        std::cout << "Outputs:";
        outfile1<<"--output-name ";
        for(auto element :  graphs_outputs[i])
        {
            std::cout<<element.name<<"; size:";
            for(auto Size : element.shape)
            {std::cout<<Size<<" ";}
            outfile1<<element.name<<";";
        }
        outfile1<<std::endl;
        std::cout << std::endl;
        std::cout <<  " The predecessors of "<<  (i>=sub1_size?sub2_type:sub1_type)<<"subgraph"<<(i>=sub1_size?(i-sub1_size):i)<<": ";
        for(auto element : predecessors_Subgraphs[i])
        {
            std::cout <<  (element>=sub1_size?sub2_type:sub1_type)<<"subgraph"<<(element>=sub1_size?(element-sub1_size):element) <<"; ";
        }
            std::cout <<std::endl;
        std::cout <<  " The successors of "<<  (i>=sub1_size?sub2_type:sub1_type)<<"subgraph"<<(i>=sub1_size?(i-sub1_size):i)<<": ";
        for(auto element : successors_Subgraphs[i])
        {
             std::cout <<  (element>=sub1_size?sub2_type:sub1_type)<<"subgraph"<<(element>=sub1_size?(element-sub1_size):element) <<"; ";
        }
            std::cout <<std::endl;
    }
    outfile1.close();
    //std::cout<<Subgraphs[0].node(0).name()<<std::endl;
///////////////////////

    // 生成粗颗粒度的cut指令
    // Print the elements of the set
    for (const auto& tensor : IOvalueNames) {
        std::cout << "Name: " << tensor.name << ", Shape: [";
        for (size_t i = 0; i < tensor.shape.size(); ++i) {
            std::cout << tensor.shape[i];
            if (i < tensor.shape.size() - 1) {
                std::cout << ", ";
            }
        }
        std::cout << "]" << std::endl;
    }

    switch (d.getType()) {
        case DeviceType::Licheepi:{
            if (strategy == SPILTE_CPU_STRUCTURE_FIRST) {
                d.GenerateCutInstruction(Subgraphs, "cpu", subgraphs_1_inputs, subgraphs_1_outputs);
                d.GenerateCutInstruction(otherSubgraphs, "npu", subgraphs_2_inputs, subgraphs_2_outputs);
            } else if (strategy == SPILTE_NPU_STRUCTURE_FIRST) {
                d.GenerateCutInstruction(Subgraphs, "npu", subgraphs_1_inputs, subgraphs_1_outputs);
                d.GenerateCutInstruction(otherSubgraphs, "cpu", subgraphs_2_inputs, subgraphs_2_outputs);
            }
            break;
        }
        default:
            std::cout << "Unknown device type" << std::endl;
            exit(0);
    }

    // 接下来应该是细致调节
    ////////////////
    
    ///////////////////
}
