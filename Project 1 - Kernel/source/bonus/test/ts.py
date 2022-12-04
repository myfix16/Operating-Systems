import argparse, os, subprocess, glob, time, re

def match_output(l1,l2):
    if len(l1)!=len(l2): return False
    # re.findall(r'[0-9].\*?[({\[]*[A-Za-z]+[)}\]].')
    ret1,ret2 = [],[]
    for i in range(len(l1)):
        ret1 += re.findall(r'[0-9]*[*]?[({\[]*(?:test2022|halt)[)}\]]*',l1[i])
        ret2 += re.findall(r'[0-9]*[*]?[({\[]*(?:test2022|halt)[)}\]]*',l2[i])
    return set(ret1)==set(ret2)

def remove_cheat(path):
    f = open(path,'r')
    a = ['execve','execl','execlp','execle','execv','execvp',"system('pstree');",'system("pstree")',"system("]
    lst = []
    for line in f:
        for word in a:
            if word in line:
                print("found cheating line:",line,end ="")
                line = line.replace(line,'')
        lst.append(line)
    f.close()
    f = open(path,'w')
    for line in lst:
        f.write(line)
    f.close()

def kill_test_proc():
    os.system("kill -9 $(pgrep halt) > /dev/null 2>&1")

def make_test_proc():
    os.system("make clean > /dev/null && make > /dev/null")

def spin_test_proc():
    subprocess.Popen(['/tmp/test2022'])
    time.sleep(1)
    return

def detect_cheat():
    sources = glob.glob("*.c")+glob.glob("*.cpp")+glob.glob("*.h")+glob.glob("*.hpp")
    for source in sources: remove_cheat(source)

def make_bonus():
    ret = os.system('''make > /dev/null 2>&1''')
    if ret:
        print("failed to make.")
    return ret # non zero indicates failure

def test_bonus():
    os.system('''cp build/pstree ./pstree''')
    os.system('''./pstree | grep 'test2022\|halt' > /tmp/o2''')
    os.system('''pstree -l | grep 'test2022\|halt'> /tmp/o1''')
    os.system('''pstree | grep 'test2022\|halt'> /tmp/o3''')
    os.system('''rm -f ./pstree''')
    with open("/tmp/o1","r") as f: long_res = f.readlines()
    with open("/tmp/o3","r") as f: short_res = f.readlines()
    try:
        with open("/tmp/o2","r") as f: stu_res = f.readlines()
    except OSError as e:
        print("failed to execute")
        return -1
    if match_output(long_res,stu_res) or match_output(short_res,stu_res): 
        print("test passed.")
    else: 
        print("correct ouputs:")
        for line in long_res: print(line)
        print("----------------------------")
        print("your ouputs:")
        for line in stu_res: print(line)
        print("----------------------------")
        print("test failed.")
        
    os.system('''rm -f /tmp/o2 /tmp/o1 > /dev/null 2>&1''')



def main(path):
    old_dir = os.getcwd()
    make_test_proc()
    spin_test_proc()
    try: os.chdir(path)
    except:
        print("invalid path")
        kill_test_proc()
        os.system("make clean > /dev/null")
        return -1
    try:
        detect_cheat()
        make_bonus()
        test_bonus()
    except Exception as e: print(e)
    os.chdir(old_dir)
    kill_test_proc()
    os.system("make clean > /dev/null")
    return 0

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Bonus test helper.')
    parser.add_argument("path")
    args = parser.parse_args()
    main(str(args.path))


