#include "partition.h"
#include <algorithm>
///////////7.11
int DetermineStructure(const onnx::GraphProto& graph, Device &d,PartitionStrategy strategy) 
{
    //std::cout <<"enabled structures:";
    //把graph节点和device的op类型逐个比对，超过3个则输出到文件中，格式与structure相同
    int node_index = 0;
    int match_count = 0;
    int structure_index = 0;
    std::vector<std::vector<std::string>> enabled_structure;
    std::vector<std::string> structure_temp;
    while(node_index < graph.node_size())
    {
        std::vector<std::string> support_op;
        const auto& node = graph.node(node_index);
        switch (strategy)
        {
            case SPILTE_CPU_STRUCTURE_FIRST:
            {
                support_op =d.getCPUSupportOp();
                break;
            }
            case SPILTE_NPU_STRUCTURE_FIRST:
            {
                support_op =d.getNPUSupportOp();
                break;
            }
            default: 
            {break;}
        }
    if(std::find(support_op.begin(),support_op.end(),node.op_type())!=support_op.end())
            {
                auto op_index=std::find(support_op.begin(),support_op.end(),node.op_type());
                structure_temp.push_back(*op_index);
            }
            else
            {
                if(structure_temp.size()>=3)
                {
                    bool isequal=0;
                    for(const auto& structure : enabled_structure)//如果这个structure尚不存在，加入structure中
                    
                    {
                        if(std::equal(structure.begin(),structure.end(),structure_temp.begin(),structure_temp.end()))
                        {
                            isequal=1;
                            break;
                        }
                    }
                    if(isequal==0)
                    {
                        enabled_structure.push_back(structure_temp);
                    }
                }
                if(structure_temp.size()!=0){
                    structure_temp.clear();
                }
                
            }
        node_index++;
    }
    
        for(const auto& structure : enabled_structure)
        {
            std::cout<<"{";
            for(const auto& op : structure)
            {
                std::cout <<"\""<< op << "\",";
            }
            std::cout<<"},"<<std::endl;
        }
    return 0;
}
// ////////////7.19
// int DetermineStructure_auto(const onnx::GraphProto& graph, Device &d,PartitionStrategy strategy) 
// {
//     //std::cout <<"enabled structures:";
//     //把graph节点和device的op类型逐个比对，超过3个则输出到文件中，格式与structure相同
//     int node_index = 0;
//     int match_count = 0;
//     int structure_index = 0;
//     std::vector<std::vector<std::string>> enabled_structure;
//     std::vector<std::string> structure_temp;
//     std::vector<onnx::NodeProto> structure_with_node_temp;
//     while(node_index < graph.node_size())
//     {
//         const auto& node = graph.node(node_index);
//         switch (strategy)
//         {
//             case SPILTE_CPU_STRUCTURE_FIRST:
//             {
//                 auto support_op =d.getCPUSupportOp();
//                 break;
//             }
//             case SPILTE_NPU_STRUCTURE_FIRST:
//             {
//                 auto support_op =d.getNPUSupportOp();
//                 break;
//             }
//             default: break;
//         }
//     if(std::find(support_op.begin(),support_op.end(),node.op_type())!=support_op.end())
//             {
//                 auto op_index=std::find(support_op.begin(),support_op.end(),node.op_type());
//                 structure_temp.push_back(*op_index);
//                 structure_with_node_temp.push_back(node);
//             }
//             else
//             {
//                 if(structure_temp.size()>=3)
//                 {
//                     bool isequal=0;
//                     for(const auto& structure : enabled_structure)//如果这个structure尚不存在，加入structure中
//                     {
//                         if(std::equal(structure.begin(),structure.end(),structure_temp.begin(),structure_temp.end()))
//                         {
//                             isequal=1;
//                             break;
//                         }
//                     }
//                     for(int j=0; j<structure_with_node_temp.size()-1; j++)
//                     {
//                         auto & node=structure_with_node_temp[j];
//                         for(auto & output_name : node.output())
//                         {
//                             if(find(structure_with_node_temp[j+1].input().begin(),structure_with_node_temp[j+1].input().end(),output_name)==structure_with_node_temp[j+1].input().end())
//                             {
//                                 split_structure();////////////
//                             }
//                         }
//                     }
//                     if(isequal==0)
//                     {
//                         enabled_structure.push_back(structure_temp);
//                     }
//                 }
//                 if(structure_temp.size()!=0){
//                     structure_temp.clear();
//                 }
                
//             }
//         node_index++;
//     }
    
//         for(const auto& structure : enabled_structure)
//         {
//             std::cout<<"{";
//             for(const auto& op : structure)
//             {
//                 std::cout <<"\""<< op << "\",";
//             }
//             std::cout<<"},"<<std::endl;
//         }
//     return 0;
// }
// std::vector<std::vector<onnx::NodeProto>> split_structure(std::vector<onnx::NodeProto> structure_with_node_temp)
// {
//     int flag = 0;
//     for(int j=0; j<structure_with_node_temp.size()-1; j++)//在每个节点处遍历断点
//     {
//         auto & node=structure_with_node_temp[j];
//         for(auto & output_name : node.output())
//         {
//             if(find(structure_with_node_temp[j+1].input().begin(),structure_with_node_temp[j+1].input().end(),output_name)!=structure_with_node_temp[j+1].input().end())
//             {
//                 //在每个断点处标记，遍历结束以后统一切分，返回切出来的二维vector 
//                 //如果有一个找到了，那就结束
//                 flag = 1;
//                 break;
//             }
//         }

//     }
// }
