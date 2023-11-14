import javalang
import csv
import os
from javalang import tree
import pandas as pd

projects = [
    'ant',
    'camel',
    'ivy',
    'jedit',
    'log4j',
    'lucene',
    'poi',
    'synapse',
    'xalan',
    'xerces'
]

versions = {
    'ant': ['1.5', '1.6', '1.7'],
    'camel': ['1.2', '1.4', '1.6'],
    'ivy': ['1.4', '2.0'],
    'jedit': ['3.2', '4.0', '4.1'],
    'log4j': ['1.0', '1.1'],
    'lucene': ['2.0', '2.2', '2.4'],
    'poi': ['1.5', '2.5', '3.0'],
    'synapse': ['1.0', '1.1', '1.2'],
    'xalan': ['2.4', '2.5'],
    'xerces': ['1.2', '1.3']
}

dict_dir = {"ant": ["src/main"], "camel": ["camel-core/src/main/java"],
            "ivy": ["src/java"], "jedit": [""], "log4j": ["src/java"],
            "lucene": ["src/java"], "poi": ["src/java"],
            "synapse": ["modules/core/src/main/java"], "velocity": ["src/java"],
            "xalan": ["src"], "xerces": ["src"]}


def parse_ast(ast_path):
    data = open(ast_path, encoding='utf-8').read()
    cu_tree = javalang.parse.parse(data)
    res = []
    for _, node in cu_tree:
        pattern = javalang.tree.ReferenceType
        if isinstance(pattern, type) and isinstance(node, pattern):
            res.append('ReferenceType_' + node.name)
        pattern = javalang.tree.MethodInvocation
        if isinstance(pattern, type) and isinstance(node, pattern):
            res.append('MethodInvocation_' + node.member)
        pattern = javalang.tree.MethodDeclaration
        if isinstance(pattern, type) and isinstance(node, pattern):
            res.append('MethodDeclaration_' + node.name)
        pattern = javalang.tree.TypeDeclaration
        if isinstance(pattern, type) and isinstance(node, pattern):
            res.append('TypeDeclaration_' + node.name)
        pattern = javalang.tree.ClassDeclaration
        if isinstance(pattern, type) and isinstance(node, pattern):
            res.append('ClassDeclaration_' + node.name)
        pattern = javalang.tree.EnumDeclaration
        if isinstance(pattern, type) and isinstance(node, pattern):
            res.append('EnumDeclaration_' + node.name)
        pattern = javalang.tree.IfStatement
        if isinstance(pattern, type) and isinstance(node, pattern):
            res.append("if_statement")
        pattern = javalang.tree.WhileStatement
        if isinstance(pattern, type) and isinstance(node, pattern):
            res.append("while_statement")
        pattern = javalang.tree.DoStatement
        if isinstance(pattern, type) and isinstance(node, pattern):
            res.append("do_statement")
        pattern = javalang.tree.ForStatement
        if isinstance(pattern, type) and isinstance(node, pattern):
            res.append("for_statement")
        pattern = javalang.tree.AssertStatement
        if isinstance(pattern, type) and isinstance(node, pattern):
            res.append("assert_statement")
        pattern = javalang.tree.BreakStatement
        if isinstance(pattern, type) and isinstance(node, pattern):
            res.append("break_statement")
        pattern = javalang.tree.ContinueStatement
        if isinstance(pattern, type) and isinstance(node, pattern):
            res.append("continue_statement")
        pattern = javalang.tree.ReturnStatement
        if isinstance(pattern, type) and isinstance(node, pattern):
            res.append("return_statement")
        pattern = javalang.tree.ThrowStatement
        if isinstance(pattern, type) and isinstance(node, pattern):
            res.append("throw_statement")
        pattern = javalang.tree.SynchronizedStatement
        if isinstance(pattern, type) and isinstance(node, pattern):
            res.append("synchronized_statement")
        pattern = javalang.tree.TryStatement
        if isinstance(pattern, type) and isinstance(node, pattern):
            res.append("try_statement")
        pattern = javalang.tree.SwitchStatement
        if isinstance(pattern, type) and isinstance(node, pattern):
            res.append("switch_statement")
        pattern = javalang.tree.BlockStatement
        if isinstance(pattern, type) and isinstance(node, pattern):
            res.append("block_statement")
        pattern = javalang.tree.StatementExpression
        if isinstance(pattern, type) and isinstance(node, pattern):
            res.append("statement_expression")
        pattern = javalang.tree.TryResource
        if isinstance(pattern, type) and isinstance(node, pattern):
            res.append("try_resource")
        pattern = javalang.tree.CatchClause
        if isinstance(pattern, type) and isinstance(node, pattern):
            res.append("catch_clause")
        pattern = javalang.tree.CatchClauseParameter
        if isinstance(pattern, type) and isinstance(node, pattern):
            res.append("catch_clause_parameter")
        pattern = javalang.tree.SwitchStatementCase
        if isinstance(pattern, type) and isinstance(node, pattern):
            res.append("switch_statement_case")
        pattern = javalang.tree.ForControl
        if isinstance(pattern, type) and isinstance(node, pattern):
            res.append("for_control")
        pattern = javalang.tree.EnhancedForControl
        if isinstance(pattern, type) and isinstance(node, pattern):
            res.append("enhanced_for_control")

    # return ' '.join(res)
    return res


if __name__ == "__main__":
    for project in projects:
        for version in versions[project]:
            # csv_data = csv.reader(open("dataset/bug-data/{}/{}.csv".format(project, project + '-' + version), 'r'))
            # next(csv_data)
            csv_path = "dataset/bug-data/{}/{}.csv".format(project, project + '-' + version)
            df = pd.read_csv(csv_path, skiprows=0)
            java_path = df.iloc[:, 0]
            all_count = 0
            wrong_count = 0
            max_len = 0
            corpus_file = open('data/tokens_file/{}_{}.txt'.format(project, version), 'w+', encoding='utf-8')
            for i in range(java_path.shape[0]):
                file_name = java_path[i].replace(".", "/") + ".java"
                # print(file_name)
                all_count += 1
                for pos_dir in dict_dir[project]:
                    path = "dataset/source-data/{}/{}/{}".format(project, project + '-' + version, pos_dir)
                    file_path = path + "/" + file_name
                    if os.path.exists(file_path):
                        break
                else:
                    df.drop(i, inplace=True)
                    wrong_count += 1
                    continue
                try:
                    file_sequence = parse_ast(file_path)
                    if len(file_sequence) > max_len:
                        max_len = len(file_sequence)
                    for w in file_sequence:
                        corpus_file.write(w + ' ')
                    corpus_file.write('\n')
                    # corpus_file.write(file_sequence + " ")
                    # sequence_and_label_file.write(str({"sequence": file_sequence, "bug": line[-1]}) + "\n")
                except Exception as e:
                    df.drop(i, inplace=True)
                    wrong_count += 1
                    # print("{}Parse Error!".format(file_path))
            print("project {}-{} all:{} fail:{}".format(project, version, all_count, wrong_count))
            df["bug"] = df["bug"].apply(lambda x: 1 if x != 0 else 0)
            df.to_csv("data/label_file/{}-{}.csv".format(project, version), index=False)
