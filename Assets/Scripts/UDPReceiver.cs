using UnityEngine;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;

public class UDPReceiver : MonoBehaviour
{
    UdpClient client;
    Thread receiveThread;
    volatile string latestData;
    Rigidbody rb;
    Vector3 startPosition;
    Vector3 initialSize;
    bool isThrown = false;
    bool launchApplied = false;
    void Start()
    {
        initialSize = transform.localScale;
        startPosition = transform.position;
        rb = GetComponent<Rigidbody>();
        client = new UdpClient(5005); 
        receiveThread = new Thread(new ThreadStart(ReceiveData));
        receiveThread.IsBackground = true;
        receiveThread.Start();
    }
    void ReceiveData()
    {
        IPEndPoint remoteEndPoint = new IPEndPoint(IPAddress.Any, 0);

        while (true)
        {
            byte[] data = client.Receive(ref remoteEndPoint);
            latestData = Encoding.UTF8.GetString(data);    
        }
    }

    void Update()
    {
     if (latestData != null)
     {
         string[] parts = latestData.Split(',');
         if (parts.Length == 9)
        {
            if (isThrown)
            {
                latestData = null;
                return;
            }
            float.TryParse(parts[0], out float velocity);
            float.TryParse(parts[1], out float scale_Factor); 
            float.TryParse(parts[2], out float posX);
            float.TryParse(parts[3], out float posY);
            float.TryParse(parts[4], out float posZ);
            Vector3 receivedPosition = new Vector3(posX, posY, posZ);
            if (!isThrown)
            {
                transform.position = receivedPosition + startPosition; // Add the start position to maintain relative movement  
            }
            float.TryParse(parts[5], out float x);
            float.TryParse(parts[6], out float y);
            float.TryParse(parts[7], out float z);
            float.TryParse(parts[8], out float w);
            Quaternion receivedRotation = new Quaternion(x, y, z, w);
            transform.rotation = receivedRotation;
            scale_Factor = Mathf.Clamp(scale_Factor, 0.1f, 10f); 
            transform.localScale = initialSize * scale_Factor;
            Debug.Log(velocity);
            latestData = null;
            if ((Mathf.Abs(velocity) > 0.01f) && !launchApplied)
            {
                rb.linearVelocity = new Vector3((-1) * velocity * 30f, rb.linearVelocity.y, rb.linearVelocity.z); 
                isThrown = true;
                launchApplied = true;
            }
        }  
     }  
    }
    void OnDestroy()
    {
    receiveThread.Abort();
    client.Close();
    }
}
