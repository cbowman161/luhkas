import os
def main():
    count = len([name for name in os.listdir('/sys/class/net') if os.path.isdir(os.path.join('/sys/class/net', name))])
    print(f'network_interface_count: {count}')
if __name__ == '__main__':
    main()