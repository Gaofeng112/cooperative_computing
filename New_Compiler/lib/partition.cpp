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
    int max_subgraph_size = d.max_subgraph_size;
	std::vector<std::string> support_op;
    std::vector<std::string> prefer_op;
	    switch(strategy) {
        case SPILTE_CPU_STRUCTURE_FIRST:{
			support_op=d.getCPUSupportOp();
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
                // if(subgraph.node_size()<=max_subgraph_size)
                // {
                //     Subgraphs.push_back(subgraph);
                // }
                // else
                // {
                //     onnx::GraphProto subgraph_new;
                //     for(int j=0;j<subgraph.node_size();j++)
                //     {
                //         if(j <= max_subgraph_size - 1)
                //         {
                //             *subgraph_new.add_node()=subgraph.node(j);
                //         }
                //         if(j > max_subgraph_size-1)
                //         {
                //             visited[sugraph_node_index[j]] = 0;
                //         }
                //     }
                //     Subgraphs.push_back(subgraph_new);
                // }
                Subgraphs.push_back(subgraph);
                //std::cout<<"subgraph "<<Subgraphs.size()<<"generated! ";
                // for(const auto& node :subgraph.node())
                // {
                //     std::cout<<node.name()<<"--";
                // }
                        
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
            // if(g.node(i).op_type()=="Constant")
            // {
            //     continue;
            // }
            int depth = 0;
            onnx::GraphProto subgraph;
            std::vector<int> sugraph_node_index;
            const auto& node=g.node(i);
            DFS_other(g,subgraph,sugraph_node_index,visited,node,i,adjacency_list, depth); 
            otherSubgraphs.push_back(subgraph);
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
    for(const auto& successor : successors_Subgraphs[index])
    {
        if(DFN[successor] == 0)//未被访问过
        {
            Tarjan(successor, rank,strongly_connected_subgraphs, DFN, LOW, stack_subgraphs, successors_Subgraphs);//访问successor
            LOW[index] = std::min(LOW[index], LOW[successor]);
        }
        else if(std::find(stack_subgraphs.begin(),stack_subgraphs.end(),successor) != stack_subgraphs.end())
        {
            LOW[index] = std::min(LOW[index], DFN[successor]);
        }
    }
    if(LOW[index] == DFN[index])//是该强连通分量子树的最小根，将其后的所有node出栈,保存得到的强连通分量
    {
        auto it = stack_subgraphs.end() - 1; 
        std:: vector<int> strongly_connected; 
        while(*it != index)
        {
            //std::cout<<*it<<"--";
            strongly_connected.insert(strongly_connected.begin(), *it);
            stack_subgraphs.pop_back();
            it = stack_subgraphs.end() - 1;
        }
        strongly_connected.insert(strongly_connected.begin(), *it);
        for(const auto& graph :strongly_connected)
        {
        }

        if(strongly_connected.size() > 1)
        {
            strongly_connected_subgraphs.push_back(strongly_connected);
        }
        stack_subgraphs.pop_back();//自身出栈

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
        std::vector<graph_adjacency_node> node_rank_list;
        calculate_node_rank_v3(g, node_rank_list);
        int index_all = 0;
        for(int scc_index = 0; scc_index < subgraph_size + other_subgraph_size; scc_index++)
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
            std::vector<graph_adjacency_node> scc_node_rank;
            int index_316_flag = 0;
            for(int i=0; i< scc_graph.node_size(); i++)
            {
                // if(scc_graph.node(i).name() == "/blocks.2/blocks.2.8/spatial_block/conv1/fn/Shape_2")
                // {
                //     index_316_flag = 1;
                // }
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
            // if(index_316_flag == 1)/* */
            // {
            //     std::cout<<"node rank: ";
            //     for(int i=0; i<scc_node_rank.size(); i++)
            //     {
            //         std::cout<<scc_node_rank[i].name<<":"<<scc_node_rank[i].rank<<"-";
            //     }
            //     std::cout<<std::endl;
            //     std::cout<<"cut rank: ";
            //     for(int i=0; i<cut_rank.size(); i++)
            //     {
            //         std::cout<<cut_rank[i]<<"-";
            //     }
            //     std::cout<<std::endl;
            //     std::exit(0);
            // }
            ////////////
            onnx::GraphProto temp_graph_upper;
            int node_in_upper = 0;
            for(int i=0; i<scc_graph.node_size(); i++)
            {
                if(scc_node_rank[i].rank < cut_rank[0])
                {
                   // *temp_graph_upper.add_node() = scc_graph.node(i);
                    node_in_upper++;
                }
            }
            int node_in_upper_added = 0;
            std::vector<onnx::GraphProto> temp_graph_upper_adder_list;
            int record_i = 0;
            std::cout<<"node size: "<<scc_graph.node_size()<<std::endl;
            std::cout<<"node in upper: "<<node_in_upper<<std::endl;
            while(node_in_upper_added < node_in_upper)
            {
                onnx::GraphProto temp_graph_upper_adder;
                for(int i=record_i; i<scc_graph.node_size(); i++)
                {
                    std::cout<<"for loop start: i= "<<i<<std::endl;
                    int i_minus_1 = 0;
                    if(i == 0)
                    {
                        i_minus_1 = 0;
                    }
                    else
                    {
                        i_minus_1 = i - 1;
                    }
                    if(scc_node_rank[i].rank < cut_rank[0]&&(i == record_i||(scc_node_rank[i].rank == scc_node_rank[i_minus_1].rank + 1)))
                    {
                        *temp_graph_upper_adder.add_node() = scc_graph.node(i);
                        node_in_upper_added ++;
                    }
                    else
                    {
                        if(scc_node_rank[i].rank >= cut_rank[0])
                        {record_i = i + 1;}
                        else{record_i = i;}
                        if(temp_graph_upper_adder.node_size() > 0)
                        {
                            temp_graph_upper_adder_list.push_back(temp_graph_upper_adder);
                            temp_graph_upper_adder.clear_node();
                        }
                        break;
                    }
                    if(i == scc_graph.node_size() - 1 && temp_graph_upper_adder.node_size()>0)
                    {
                        temp_graph_upper_adder_list.push_back(temp_graph_upper_adder);
                        temp_graph_upper_adder.clear_node();
                    }                    
                }
                std::cout<<"loop ended:temp graph upper adder size: "<<temp_graph_upper_adder.node_size()<<" "<<record_i<<"/"<<scc_graph.node_size()<<" node_in_upper_added:"<<node_in_upper_added<<std::endl;
            }
            if(scc_index < subgraph_size)
            {
                Subgraphs[scc_index] = temp_graph_upper_adder_list[0];
            }
            else
            {
                otherSubgraphs[scc_index - subgraph_size] = temp_graph_upper_adder_list[0];
            }

            if(temp_graph_upper_adder_list.size() > 1)
            {
                for(int i = 1; i< temp_graph_upper_adder_list.size(); i++)
                {
                    if(scc_index < subgraph_size)
                    {
                        Subgraphs.push_back(temp_graph_upper_adder_list[i]);
                    }
                    else
                    {
                        otherSubgraphs.push_back(temp_graph_upper_adder_list[i]);
                    }                
                }                
            }
            std::cout<<"scc index"<<scc_index<<" scc size: "<<scc_graph.node_size()<<std::endl;
            std::cout<<"scc node rank: ";
            for(int i=0; i< scc_graph.node_size(); i++)
            {
                std::cout<<scc_node_rank[i].name<<" "<<scc_node_rank[i].rank<<" ";
            }
            std::cout<<std::endl;
            // if(scc_index < subgraph_size)
            // {
            //     Subgraphs[scc_index] = temp_graph_upper;
            // }
            // else
            // {
            //     otherSubgraphs[scc_index - subgraph_size] = temp_graph_upper;
            // }
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
onnx::GraphProto determinegraphtype_v2(
    int index,
    std::vector<onnx::GraphProto>& Subgraphs,
    std::vector<onnx::GraphProto>& otherSubgraphs,
    int subgraph_size
)
{
    if(index < subgraph_size)
    {
        return Subgraphs[index];
    }
    else
    {
        return otherSubgraphs[index - subgraph_size];
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
void find_subgraph_pair_v2(
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
            scc_graphs_inputs.push_back(graph_inputs);
            scc_graphs_outputs.push_back(graph_outputs);
        }
        int find_flag=0;
        std::vector<std::vector<int>> scc_pairs;
        std::vector<int> is_pushed;
        for(int j = 0; j < strongly_connected.size(); j++)
        {
            is_pushed.push_back(0);
        }
        for(int i = 0; i < strongly_connected.size(); i++)
        {
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
                                is_pushed[i]=1;
                                find_flag=1;
                                break;
                            }
                        }
                    }
                    if(is_pushed[i]==1)
                    {
                        break;
                    }
                }
                if(is_pushed[i]==1)
                {
                    break;
                }
            }
        }
        if(scc_pairs.size() != 0)
        {
            sccs_pairs.push_back(scc_pairs);
        }
    count ++;
    }
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
    std::vector<onnx::GraphProto>& scc_pair_cut,
    int subgraph_size
)
{
    std::vector<graph_adjacency_node> pair_node_list = calculate_node_rank(scc_pair, Subgraphs,otherSubgraphs);
    int master_graph = 0;
    for(const auto& node : pair_node_list)
    {
        if(node.rank==0)
        {
            int find_flag = -1;
            onnx::GraphProto graph_temp = determinegraphtype_v2(scc_pair[0],Subgraphs, otherSubgraphs,subgraph_size);
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
                break;
            }
            else{master_graph = 1;break;}
        }
    }
    int slave_graph = 1 - master_graph;
    //找到master与slave图相接的位置
    int cut_rank = -1;
    for(const auto& output : graphs_outputs[scc_pair[slave_graph]])
    {
        for(const auto& input : graphs_inputs[scc_pair[master_graph]])
        {
            
            if(input.name ==output.name)
            {
                int node_index = 0;
                onnx::GraphProto graph_temp = determinegraphtype_v2(scc_pair[slave_graph],Subgraphs, otherSubgraphs,subgraph_size);
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
                                }
                            }
                            else
                            {
                                onnx::GraphProto graph_temp_1 = determinegraphtype_v2(scc_pair[master_graph], Subgraphs, otherSubgraphs,subgraph_size);
                                if(cut_rank==-1||cut_rank>pair_node_list[node_index+ graph_temp_1.node_size()].rank)
                                {
                                    cut_rank = pair_node_list[node_index+ graph_temp_1.node_size()].rank; 
                                }
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
    onnx::GraphProto graph_temp = determinegraphtype_v2(scc_pair[master_graph],Subgraphs, otherSubgraphs,subgraph_size);
    int master_graph_size = graph_temp.node_size();
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
            onnx::GraphProto graph_temp_2 = determinegraphtype_v2(scc_pair[slave_graph],Subgraphs, otherSubgraphs,subgraph_size);
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
    scc_pair_cut.push_back(determinegraphtype_v2(scc_pair[slave_graph],Subgraphs, otherSubgraphs,subgraph_size));
    if(master_graph == 1)
    {
        int temp = scc_pair[0];
        scc_pair[0] = scc_pair[1];
        scc_pair[1] = temp;
        master_graph = 0;
    }//master一定是0
    std::vector<int> return_value;
    return_value.push_back(master_graph);
    return_value.push_back(cut_rank);
    return return_value;
}
void eliminate_pair_v2(
    std::vector<onnx::GraphProto>& Subgraphs,
    std::vector<onnx::GraphProto>& otherSubgraphs,
    std::vector<std::unordered_set<NodeTensor>>& graphs_inputs,
    std::vector<std::unordered_set<NodeTensor>>& graphs_outputs,
    std::vector<std::vector<int>>& strongly_connected_subgraphs,
    int subgraph_size
)
{
    int original_node_size = 0;
    for(auto& subgraph : Subgraphs)
    {
        original_node_size += subgraph.node_size();
    }
    for(auto& subgraph : otherSubgraphs)
    {
        original_node_size += subgraph.node_size();
    }
    int othergraph_size = otherSubgraphs.size();
    std::vector<std::vector<std::vector<int>>> sccs_pairs;
    find_subgraph_pair_v2(strongly_connected_subgraphs, Subgraphs, otherSubgraphs, graphs_inputs, graphs_outputs, sccs_pairs);    
    for(auto& scc_pairs : sccs_pairs)
    {
        for(auto& scc_pair : scc_pairs)
        {
            std::vector<onnx::GraphProto> scc_pair_cut;
            cut_pair(Subgraphs, otherSubgraphs, graphs_inputs, graphs_outputs, scc_pair, scc_pair_cut, subgraph_size);
            if(scc_pair[0] < subgraph_size)
            {
                Subgraphs[scc_pair[0]] = scc_pair_cut[0];
                Subgraphs.push_back(scc_pair_cut[1]);
            }
            else
            {
                otherSubgraphs[scc_pair[0]-subgraph_size] = scc_pair_cut[0];
                otherSubgraphs.push_back(scc_pair_cut[1]);
            }

            if(scc_pair[1] < subgraph_size)
            {
                Subgraphs[scc_pair[1]] = scc_pair_cut[2];
            }
            else
            {
                otherSubgraphs[scc_pair[1]-subgraph_size] = scc_pair_cut[2];
            }            
        }
    }
}
void cut_crossed_pair(
    std::vector<onnx::GraphProto>& Subgraphs,
    std::vector<onnx::GraphProto>& otherSubgraphs,
    std::vector<std::unordered_set<NodeTensor>>& graphs_inputs,
    std::vector<std::unordered_set<NodeTensor>>& graphs_outputs,
    std::vector<std::vector<int>> scc_pairs_crossed,
    std::vector<std::vector<onnx::GraphProto>>& scc_pair_cut_final,
    int subgraph_size
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
        std::vector<int> return_value = cut_pair(Subgraphs, otherSubgraphs, graphs_inputs, graphs_outputs, scc_pair, scc_pair_cut, subgraph_size);  
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
    std::vector<std::vector<int>>& strongly_connected_subgraphs,
    int subgraph_size
)
{
    int original_node_size = 0;
    for(auto& subgraph : Subgraphs)
    {
        original_node_size += subgraph.node_size();
    }
    for(auto& subgraph : otherSubgraphs)
    {
        original_node_size += subgraph.node_size();
    }
    int othergraph_size = otherSubgraphs.size();
    std::vector<std::vector<std::vector<onnx::GraphProto>>> sccs_pairs_cut_multi;
    std::vector<std::vector<std::vector<int>>> sccs_pairs;
    find_subgraph_pair(strongly_connected_subgraphs, Subgraphs, otherSubgraphs, graphs_inputs, graphs_outputs, sccs_pairs);
    for(auto &scc_pairs : sccs_pairs)
    {
        std::vector<std::vector<onnx::GraphProto>> scc_pairs_cut;
        std::vector<std::vector<std::vector<int>>> scc_crossed_pairs_multi;
        find_crossed_pair(scc_pairs, scc_crossed_pairs_multi);
        for(auto &scc_pair : scc_pairs)
        {
            std::vector<onnx::GraphProto> scc_pair_cut;
            cut_pair(Subgraphs, otherSubgraphs, graphs_inputs, graphs_outputs, scc_pair, scc_pair_cut,subgraph_size);
            scc_pairs_cut.push_back(scc_pair_cut);
        }
        for(auto &scc_crossed_pairs : scc_crossed_pairs_multi)
        {
            std::vector<std::vector<onnx::GraphProto>> scc_pairs_cut_multi;
            cut_crossed_pair(Subgraphs, otherSubgraphs, graphs_inputs, graphs_outputs, scc_crossed_pairs, scc_pairs_cut_multi,subgraph_size);
            scc_pairs_cut.insert(scc_pairs_cut.end(),scc_pairs_cut_multi.begin(), scc_pairs_cut_multi.end());
            scc_pairs.insert(scc_pairs.end(),scc_crossed_pairs.begin(), scc_crossed_pairs.end());
        }
        sccs_pairs_cut_multi.push_back(scc_pairs_cut);
    }
    //int subgraph_size = Subgraphs.size(); 
    std::cout<<"size"<<subgraph_size<<" "<<othergraph_size<<std::endl;
    for(int i = sccs_pairs.size() - 1; i >= 0; i--)
    {
        for(int j = sccs_pairs[i].size() - 1; j >= 0 ; j--)
        {
            if(sccs_pairs_cut_multi[i][j].size() > 1)
            {         
                int original_node_size_sub = 0;      
                if(sccs_pairs[i][j][0] < subgraph_size)
                {
                    original_node_size_sub = Subgraphs[sccs_pairs[i][j][0]].node_size();
                    Subgraphs[sccs_pairs[i][j][0]] = sccs_pairs_cut_multi[i][j][0];
                    Subgraphs.push_back(sccs_pairs_cut_multi[i][j][1]);
                }
                else
                {
                    original_node_size_sub = otherSubgraphs[sccs_pairs[i][j][0] - subgraph_size].node_size();
                    otherSubgraphs[sccs_pairs[i][j][0] - subgraph_size] = sccs_pairs_cut_multi[i][j][0];
                    otherSubgraphs.push_back(sccs_pairs_cut_multi[i][j][1]);
                }
                ////10.31
                if(sccs_pairs_cut_multi[i][j].size() > 2)
                {
                    if(sccs_pairs[i][j][1] < subgraph_size)
                    {
                        Subgraphs[sccs_pairs[i][j][1]] = sccs_pairs_cut_multi[i][j][2];
                    }
                    else
                    {
                        otherSubgraphs[sccs_pairs[i][j][1] - subgraph_size] = sccs_pairs_cut_multi[i][j][2];
                    }
                }
                ////10.31end
                int node_size_after_cut = 0;
                for(auto& subgraph : Subgraphs)
                {
                    node_size_after_cut += subgraph.node_size();
                }
                for(auto& subgraph : otherSubgraphs)
                {
                    node_size_after_cut += subgraph.node_size();
                }
                if(node_size_after_cut != original_node_size)
                {
                    std::cout<<"error : node size after cut is not equal to original node size after cut pair "<<i<<std::endl;
                    
                    std::exit(1);
                }
            }
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

void handle_onnx_error(std::vector<onnx::GraphProto> &Subgraphs, std::vector<onnx::GraphProto> &otherSubgraphs, const onnx::GraphProto &g)
{
    std::unordered_set<NodeTensor> output_nodes = getOutvalue(g);
    for(auto& node : output_nodes)
    {
        onnx::GraphProto errorgraph1, errorgraph2;
        int error_flag = 0;
        int find_flag = 0;
        int error1_index = 0;
        int error2_index = 0;
        for(auto& graph : Subgraphs)
        {
            for(auto& input : graph.input())
            {
                if(input.name() == node.name)
                {
                    errorgraph1 = graph;
                    error_flag = 1;
                    break;
                }
            }
            if(error_flag == 1)
            {
                break;
            }
            error1_index++;
        }
        if(error_flag == 0)
        {
            error1_index = 0;
            for(auto& graph : otherSubgraphs)
            {
                for(auto& input : graph.input())
                {
                    if(input.name() == node.name)
                    {
                        errorgraph1 = graph;
                        error_flag = 2;
                        break;
                    }
                }
                if(error_flag == 2)
                {
                    break;
                }
                error1_index++;
            }
        }
        if(error_flag > 0)
        {
            for(auto& graph : Subgraphs)
            {
                for(auto& output : graph.output())
                {
                    if(output.name() == node.name)
                    {
                        errorgraph2 = graph;
                        find_flag = 1;
                        break;
                    }
                }
                {
                    if(find_flag == 1)
                    {
                        break;
                    }
                }
                error2_index++;
            }
            if(find_flag == 0)
            {
                error2_index = 0;
                for(auto& graph : otherSubgraphs)
                {
                    for(auto& input : graph.input())
                    {
                        if(input.name() == node.name)
                        {
                            errorgraph2 = graph;
                            find_flag = 2;
                            break;
                        }
                    }
                    if(find_flag == 2)
                    {
                        break;
                    }
                    error2_index++;
                }
            }
        }
        if(find_flag == 1 && error_flag == 1|| find_flag == 2)
        {
            mergeGraphs(errorgraph2, errorgraph1);
            Subgraphs.erase(Subgraphs.begin()+error1_index);
        }
        else if(error_flag == 2)
        {
            mergeGraphs(errorgraph1, errorgraph2);
            otherSubgraphs.erase(otherSubgraphs.begin()+error2_index);
        }
    }

}
void Partition::PartitionGraph(const onnx::GraphProto &g, Device& d, PartitionStrategy strategy, const std::unordered_map<std::string, NodeIOSize> &node_io_size) {
    //std::unordered_set<NodeTensor> initializerNames = getInitializer(g);
    std::unordered_set<NodeTensor> IOvalueNames = getIOvalue(g);
    int* visited = (int*)malloc(g.node_size()*sizeof(int));
    std::vector<graph_adjacency_node> adjacency_list=get_adjancency_list(g, visited);
    std::vector<onnx::GraphProto> otherSubgraphs;
    determine_subgraphs(g,otherSubgraphs, d, visited, adjacency_list,strategy);
    free(visited);
    std::vector<graph_adjacency_node>().swap(adjacency_list);
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
//    std::cout << "graph node size:" << graph_node_size_minus_constant << std::endl;
    std::cout << "sub node size:" << node_sum << std::endl;


    std::vector<std::unordered_set<NodeTensor>> subgraphs_1_inputs;
    std::vector<std::unordered_set<std::string>> subgraphs_1_input_nodes;
    std::vector<std::unordered_set<std::string>> subgraphs_1_nodes;
    for (const auto& sg : Subgraphs) {
        std::unordered_set<NodeTensor> graphInputs;
        determineGraphInput(sg, IOvalueNames, graphInputs);
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
    int graph_node_size_minus_constant = g.node_size();
    for(const auto& node: g.node())
    {
        if(node.op_type() == "Constant")
        {
            graph_node_size_minus_constant--;
        }
    }   
    std::cout<<"total number of nodes in subgraphs:"<<node_number<<std::endl;
    std::cout<<"total number of nodes in origional graph:"<<graph_node_size_minus_constant<<std::endl;

/////////////////////////6.29
    std::vector<std::unordered_set<NodeTensor>> graphs_inputs;
    graphs_inputs.insert(graphs_inputs.end(),subgraphs_1_inputs.begin(),subgraphs_1_inputs.end());
    graphs_inputs.insert(graphs_inputs.end(),subgraphs_2_inputs.begin(),subgraphs_2_inputs.end());
    std::vector<std::unordered_set<NodeTensor>> graphs_outputs;
    graphs_outputs.insert(graphs_outputs.end(),subgraphs_1_outputs.begin(),subgraphs_1_outputs.end());
    graphs_outputs.insert(graphs_outputs.end(),subgraphs_2_outputs.begin(),subgraphs_2_outputs.end());

    std::vector<std::vector<int>> predecessors_Subgraphs(graphs_inputs.size());
    std::vector<std::vector<int>> successors_Subgraphs(graphs_inputs.size());
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
        if(predecessors.size() == 0)
        {
            std::cout<<"subgraph "<<i<<" has no predecessors"<<std::endl;
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
    free(DFN);
    free(LOW);
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
        std::cout << "scc2:";
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
        std::cout << "scc3:";
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
    std::cout << "node_num after cut " << node_num_all << std::endl;
    if(node_num_all != g.node_size())
    {
        std::cout << "num error!" << std::endl;
        exit(0);
    }
    ////从这里添加cut pair
    int count_cut_pair = 0;
    while(1)
    {
        count_cut_pair ++;
        if(count_cut_pair > 5)
        {
            std::cout << "cut pair error! So many times!" << std::endl;
            exit(0);
        }
    int subgraph_size = Subgraphs.size();
    if(count_cut_pair == 1)
    {
        std::vector<std::vector<int>> strongly_connected_subgraphs_all;
        std::vector<int> scc_all;
        for(int i = 0; i < Subgraphs.size() + otherSubgraphs.size(); i++)
        {
            scc_all.push_back(i);
        }
        strongly_connected_subgraphs_all.push_back(scc_all);
        eliminate_pair_v2(Subgraphs, otherSubgraphs, graphs_inputs, graphs_outputs, strongly_connected_subgraphs_all, subgraph_size);
    }
    else
    {
       eliminate_pair_v2(Subgraphs, otherSubgraphs, graphs_inputs, graphs_outputs, strongly_connected_subgraphs, subgraph_size); 
    }
    handle_onnx_error(Subgraphs, otherSubgraphs, g);
    strongly_connected_subgraphs.clear();
    predecessors_Subgraphs.clear();
    successors_Subgraphs.clear();
    std::vector<std::unordered_set<NodeTensor>>().swap(subgraphs_2_inputs);
    std::vector<std::unordered_set<NodeTensor>>().swap(subgraphs_1_inputs);
    std::vector<std::unordered_set<NodeTensor>>().swap(subgraphs_2_outputs);
    std::vector<std::unordered_set<NodeTensor>>().swap(subgraphs_1_outputs);
    std::vector<std::unordered_set<NodeTensor>>().swap(graphs_inputs);
    std::vector<std::unordered_set<NodeTensor>>().swap(graphs_outputs);
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
    node_num_all = 0;
    for(const auto& sg : Subgraphs)
    {
        node_num_all += sg.node_size();
    }
    for(const auto& sg : otherSubgraphs)
    {
        node_num_all += sg.node_size();
    }
    int* DFN_4 = (int *)malloc(graphs_inputs.size() * sizeof(int));
    int* LOW_4 = (int *)malloc(graphs_inputs.size() * sizeof(int));
    for(int i = 0; i < graphs_inputs.size(); i++)
    {
        DFN_4[i] = 0;
        LOW_4[i] = 0;
    }
    temp_count = 0;
    for(const auto& predecessors : predecessors_Subgraphs)
    {
        if(DFN_[temp_count] == 0)
        {
            std::vector<int> stack_subgraphs;
            int depth = 0;
            Tarjan(temp_count, depth, strongly_connected_subgraphs, DFN_4, 
            LOW_4, stack_subgraphs, successors_Subgraphs);
        }
        temp_count ++ ;
    } 
    std::string file_name_scc4 = "scc4.txt";
    std::ofstream outfile_scc4(file_name_scc4);    
    for(const auto& scc : strongly_connected_subgraphs)
    {
        std::cout << "scc4:";
        for(const auto& scc_id : scc)
        {
            std::cout << scc_id << " ";
            outfile_scc4 << "subgraph" << scc_id << " input:";
            for(const auto& scc_input : graphs_inputs[scc_id])
            {
                outfile_scc4 << scc_input.name << ";";
            }
            outfile_scc4 << " output:";
            for(const auto& scc_output : graphs_outputs[scc_id])
            {
                outfile_scc4 << scc_output.name << ";";
            }
            outfile_scc4 << std::endl;
        }
        
        std::cout << std::endl;
    }
    outfile_scc.close();
    free(DFN_4);
    free(LOW_4);
    std::cout << "node num in original graph: " << g.node_size() << std::endl;
    std::cout << "node_num after cut " << node_num_all << std::endl;
    if(node_num_all != g.node_size())
    {
        std::cout << "num error!, time" <<count_cut_pair<< std::endl;
        exit(0);
    }
    if(strongly_connected_subgraphs.size() == 0)
    {
        break;
    }
    }//end of while
    std::string file_name_predecessor_4 = "predecessor_final_4.txt";
    std::string file_name_successor_4 = "successor_final_4.txt";
    std::ofstream outfile_predecessor_4(file_name_predecessor_4);
    std::ofstream outfile_successor_4(file_name_successor_4);
    if (!(outfile_predecessor_4.is_open()&&outfile_successor_4.is_open())) {
        std::cerr << "Error opening file." << std::endl;
        exit(0);
    }
    for(int i=0;i<graphs_inputs.size();i++)
    {
        outfile_predecessor_4 << "predecessor of subgraph " << i << ":";
        for(const auto& predecessor : predecessors_Subgraphs[i])
        {
            outfile_predecessor_4 << predecessor << ";";
        }
        outfile_predecessor_4 << std::endl;
        outfile_successor_4 << "successor of subgraph " << i << ":";
        for(const auto& successor : successors_Subgraphs[i])
        {
            outfile_successor_4 << successor << ";";
        }
        outfile_successor_4 << std::endl;
    }
    outfile_predecessor_4.close();
    outfile_successor_4.close();
    print_subgraphs(Subgraphs, "./subgraphs_final_4.txt", otherSubgraphs, "./other_subgraphs_final_4.txt");
    ////*    
    int temp_count_subgraph = 0;

    std::ofstream outfile_conv_flag("end_with_conv.txt");
    for(const auto & graph_outputs : subgraphs_1_outputs)
    {
        int find_flag = 0;
        for(const auto& graph_output : graph_outputs)
        {
            for(const auto& node : Subgraphs[temp_count_subgraph].node())
            {
                for(const auto& output : node.output())
                {
                    if(graph_output.name == output&&node.op_type()=="Conv")
                    {
                        outfile_conv_flag<<temp_count_subgraph<<" ";
                        find_flag = 1;
                        break;
                    }
                }
                if(find_flag)
                {
                    break;
                }
            }
            if(find_flag)
            {
                break;
            }
        }
        temp_count_subgraph ++;
    }
    outfile_conv_flag.close();
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
            {
                int find_flag=0;
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
            }
        }
        else
        {
            std::cout << "sort count:" <<sort_count << std::endl;
            for(int i=0; i<graphs_inputs.size();i++) ////遍历所有子图
            {
                int find_flag=0;
                if(issort_Subgraphs[i]==1&&i!=graphs_inputs.size()-1){continue;}////如果已经排过序了，跳过这个子图
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
                }
                else {order_Subgraphs[i]=sort_count+1;issort_Subgraphs[i]=0;finished_flag=0;}
                if(i==graphs_inputs.size()-1)//本循环到最后再统一加入队列，防止出现本轮循环新加入的为后面子图的前驱
                {
                    for(int j=0; j<graphs_inputs.size();j++)
                    {
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
    std::cout << "node num in original graph: " << g.node_size() << std::endl;
    std::cout << "node_num after cut " << node_num_all << std::endl;
}
