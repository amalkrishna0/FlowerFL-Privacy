import subprocess
import sys
import time

def start_processes(client_commands):
    """Starts the server and client processes in separate PowerShell windows."""
    
    # Start the server in a new PowerShell window and keep it open
    subprocess.Popen(["powershell", "-Command", "Start-Process powershell -ArgumentList 'python server.py; Read-Host \"Press Enter to exit\"'"], shell=True)

    # Wait a few seconds to ensure the server starts
    time.sleep(5)

    # Start each client in a new PowerShell window and keep it open
    for cmd in client_commands:
        subprocess.Popen(["powershell", "-Command", cmd], shell=True)

    print("All processes started in separate PowerShell windows.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python run_fl.py <option>")
        print("1 - Run with malicious client")
        print("2 - Run with noisy client")
        sys.exit(1)

    option = sys.argv[1]

    if option == "1":
        clients = [
            "Start-Process powershell -ArgumentList 'python client.py --labels 0 1 2; Read-Host \"Press Enter to exit\"'",
            "Start-Process powershell -ArgumentList 'python client.py --labels 3 4 5 --malicious; Read-Host \"Press Enter to exit\"'",
            "Start-Process powershell -ArgumentList 'python client.py --labels 6 7 8 9; Read-Host \"Press Enter to exit\"'"
        ]
    elif option == "2":
        clients = [
            "Start-Process powershell -ArgumentList 'python client.py --labels 0 1 2; Read-Host \"Press Enter to exit\"'",
            "Start-Process powershell -ArgumentList 'python client.py --labels 3 4 5 6 --noise; Read-Host \"Press Enter to exit\"'",
            "Start-Process powershell -ArgumentList 'python client.py --labels 7 8 9; Read-Host \"Press Enter to exit\"'"
        ]
    elif option == "3":
        clients = [
            "Start-Process powershell -ArgumentList 'python client.py --labels 0 1 2 3; Read-Host \"Press Enter to exit\"'",
            "Start-Process powershell -ArgumentList 'python client.py --labels 3 4 5 6; Read-Host \"Press Enter to exit\"'",
            "Start-Process powershell -ArgumentList 'python client.py --labels 6 7 8 9; Read-Host \"Press Enter to exit\"'"
        ]
    elif option == "4":
        clients = [
            "Start-Process powershell -ArgumentList 'python client.py --labels 0 1 2 3; Read-Host \"Press Enter to exit\"'",
            "Start-Process powershell -ArgumentList 'python client.py --labels 3 4 5 6 --noise; Read-Host \"Press Enter to exit\"'",
            "Start-Process powershell -ArgumentList 'python client.py --labels 6 7 8 9; Read-Host \"Press Enter to exit\"'"
        ]
    elif option == "5":
        clients = [
            "Start-Process powershell -ArgumentList 'python client.py --labels 0 1; Read-Host \"Press Enter to exit\"'",
            "Start-Process powershell -ArgumentList 'python client.py --labels 2 3; Read-Host \"Press Enter to exit\"'",
            "Start-Process powershell -ArgumentList 'python client.py --labels 4 5; Read-Host \"Press Enter to exit\"'",
            "Start-Process powershell -ArgumentList 'python client.py --labels 6 7; Read-Host \"Press Enter to exit\"'",
            "Start-Process powershell -ArgumentList 'python client.py --labels 8 9; Read-Host \"Press Enter to exit\"'",
            
        ]
    elif option == "6":
        clients = [
            "Start-Process powershell -ArgumentList 'python client.py --labels 0 1; Read-Host \"Press Enter to exit\"'",
            "Start-Process powershell -ArgumentList 'python client.py --labels 2 3 --noise; Read-Host \"Press Enter to exit\"'",
            "Start-Process powershell -ArgumentList 'python client.py --labels 4 5; Read-Host \"Press Enter to exit\"'",
            "Start-Process powershell -ArgumentList 'python client.py --labels 6 7; Read-Host \"Press Enter to exit\"'",
            "Start-Process powershell -ArgumentList 'python client.py --labels 8 9; Read-Host \"Press Enter to exit\"'",
            
        ]
    else:
        print("Invalid option! Use:")
        print("1 - Run with malicious client")
        print("2 - Run with noisy client")
        pribt("3 - Run with normal clients")
        sys.exit(1)

    start_processes(clients)
